# masking.py

# Copyright 2025 EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# [...]
from typing import List, Tuple, Dict, Any, Union, Optional
import random
from timm.models.layers import to_2tuple
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet

from .utils import to_unified_multimodal_vocab


class SimpleMultimodalMasking(object):
    def __init__(
            self,
            modalities: List[str],
            vocab_sizes: List[int],
            max_seq_lens: List[int],
            input_alphas: List[float],
            target_alphas: List[float],
            input_tokens_range: Union[int, Tuple[int, int]],
            target_tokens_range: Union[int, Tuple[int, int]],
            overlap_vocab: bool = True,
            overlap_posembs: bool = True,
            include_unmasked_data_dict: bool = False,
        ):
        self.modalities = modalities
        self.num_modalities = len(modalities)
        self.vocab_sizes = vocab_sizes
        self.max_seq_lens = max_seq_lens
        self.input_alphas = torch.tensor(input_alphas)
        self.target_alphas = torch.tensor(target_alphas)
        self.input_tokens_range = to_2tuple(input_tokens_range)
        self.target_tokens_range = to_2tuple(target_tokens_range)
        self.overlap_vocab = overlap_vocab
        self.overlap_posembs = overlap_posembs
        self.include_unmasked_data_dict = include_unmasked_data_dict

        # cumulative shifts if you don't want overlap in 1D pos embeddings
        self.max_seq_len_shifts = torch.tensor(max_seq_lens).cumsum(0) - max_seq_lens[0]

        # Dirichlet samplers
        eps = 1e-9
        self.input_dirichlet = Dirichlet(torch.clamp(self.input_alphas, min=eps))
        self.target_dirichlet = Dirichlet(torch.clamp(self.target_alphas, min=eps))

    def input_token_budget(self, num_input_tokens: int, max_tokens: torch.Tensor) -> List[int]:
        # same as before…
        input_budget = (self.input_dirichlet.sample() * num_input_tokens).floor().int()
        diff = num_input_tokens - input_budget.sum()
        input_budget += torch.bincount(
            self.input_dirichlet.sample((diff,)).argmax(-1),
            minlength=len(input_budget)
        )
        return torch.clamp(input_budget, max=max_tokens).tolist()

    def target_token_budget(
            self,
            input_token_budget: List[int],
            num_target_tokens: int,
            max_tokens: torch.Tensor,
        ) -> List[int]:
        # same as before…
        max_remaining = max_tokens - torch.tensor(input_token_budget)
        target_budget = (self.target_dirichlet.sample() * num_target_tokens).floor().int()
        diff = num_target_tokens - target_budget.sum()
        target_budget += torch.bincount(
            self.target_dirichlet.sample((diff,)).argmax(-1),
            minlength=len(target_budget)
        )
        return torch.clamp(target_budget, max=max_remaining).tolist()

    def perform_random_masking(
            self,
            data_dict: Dict[str, Any],
            input_token_budget: List[int],
            target_token_budget: List[int],
        ) -> Dict[str, Any]:
        enc_tokens, enc_pos, enc_mods, enc_frame_pos, enc_spatial_pos = [], [], [], [], []
        dec_tokens, dec_pos, dec_mods, dec_frame_pos, dec_spatial_pos = [], [], [], [], []

        # for each modality, pick which positions go to encoder vs. decoder
        for m_idx, mod in enumerate(self.modalities):
            seq = data_dict[f"{mod}_tokens"]                      # [N]
            N = seq.shape[0]
            n_in = input_token_budget[m_idx]
            n_out = target_token_budget[m_idx]

            # shuffle and split
            shuffle_idx = torch.randperm(N)
            in_idx  = shuffle_idx[:n_in].sort()[0]
            out_idx = shuffle_idx[n_in:n_in+n_out].sort()[0]

            shift = 0 if self.overlap_posembs else self.max_seq_len_shifts[m_idx]
            enc_pos.append(in_idx + shift)
            dec_pos.append(out_idx + shift)

            # tokens themselves
            enc_tokens.append(seq[in_idx])
            dec_tokens.append(seq[out_idx])

            # modality IDs
            enc_mods.append(m_idx * torch.ones(n_in, dtype=torch.long))
            dec_mods.append(m_idx * torch.ones(n_out, dtype=torch.long))

            # frame positions
            fkey = f"{mod}_frame_positions"
            if fkey in data_dict:
                enc_frame_pos.append(data_dict[fkey][in_idx])
                dec_frame_pos.append(data_dict[fkey][out_idx])
            else:
                enc_frame_pos.append(torch.zeros(n_in, dtype=torch.long))
                dec_frame_pos.append(torch.zeros(n_out, dtype=torch.long))

            # **NEW** spatial positions
            skey = f"{mod}_spatial_positions"
            if skey in data_dict:
                enc_spatial_pos.append(data_dict[skey][in_idx])
                dec_spatial_pos.append(data_dict[skey][out_idx])
            else:
                enc_spatial_pos.append(torch.zeros(n_in, dtype=torch.long))
                dec_spatial_pos.append(torch.zeros(n_out, dtype=torch.long))

        # concatenate across modalities
        enc_tokens      = torch.cat(enc_tokens)
        enc_pos         = torch.cat(enc_pos)
        enc_mods        = torch.cat(enc_mods)
        enc_frame_pos   = torch.cat(enc_frame_pos)
        enc_spatial_pos = torch.cat(enc_spatial_pos)

        dec_tokens      = torch.cat(dec_tokens)
        dec_pos         = torch.cat(dec_pos)
        dec_mods        = torch.cat(dec_mods)
        dec_frame_pos   = torch.cat(dec_frame_pos)
        dec_spatial_pos = torch.cat(dec_spatial_pos)

        # pad out to fixed lengths
        max_in, max_out = self.input_tokens_range[1], self.target_tokens_range[1]
        pad_in  = max_in  - enc_tokens .shape[0]
        pad_out = max_out - dec_tokens .shape[0]

        def _pad(x, length, val):
            return F.pad(x, (0, length), value=val)

        enc_tokens      = _pad(enc_tokens, pad_in, 0)
        enc_pos         = _pad(enc_pos, pad_in, 0)
        enc_mods        = _pad(enc_mods, pad_in, 0)
        enc_frame_pos   = _pad(enc_frame_pos, pad_in, 0)
        enc_spatial_pos = _pad(enc_spatial_pos, pad_in, 0)

        dec_tokens      = _pad(dec_tokens, pad_out, -100)
        dec_pos         = _pad(dec_pos, pad_out, 0)
        dec_mods        = _pad(dec_mods, pad_out, 0)
        dec_frame_pos   = _pad(dec_frame_pos, pad_out, 0)
        dec_spatial_pos = _pad(dec_spatial_pos, pad_out, 0)

        # build pad masks
        enc_pad_mask = torch.ones(max_in,  dtype=torch.bool)
        enc_pad_mask[-pad_in:] = False
        dec_pad_mask = torch.ones(max_out, dtype=torch.bool)
        dec_pad_mask[-pad_out:] = False

        return {
            'enc_tokens':           enc_tokens,
            'enc_positions':        enc_pos,
            'enc_modalities':       enc_mods,
            'enc_frame_positions':  enc_frame_pos,
            'enc_spatial_positions':enc_spatial_pos,   # ? new
            'enc_pad_mask':         enc_pad_mask,

            'dec_tokens':           dec_tokens,
            'dec_positions':        dec_pos,
            'dec_modalities':       dec_mods,
            'dec_frame_positions':  dec_frame_pos,
            'dec_spatial_positions':dec_spatial_pos,   # ? new
            'dec_pad_mask':         dec_pad_mask,
        }

    def __call__(self, data_dict):
        # optionally unify vocab
        if not self.overlap_vocab:
            data_dict = to_unified_multimodal_vocab(
                data_dict, self.modalities, self.vocab_sizes
            )

        max_tokens = torch.tensor(self.max_seq_lens)
        num_in  = random.randint(*self.input_tokens_range)
        num_out = random.randint(*self.target_tokens_range)

        in_budget  = self.input_token_budget(num_in,  max_tokens)
        out_budget = self.target_token_budget(in_budget, num_out, max_tokens)

        masked = self.perform_random_masking(data_dict, in_budget, out_budget)

        if self.include_unmasked_data_dict:
            masked['unmasked_data_dict'] = data_dict

        return masked

