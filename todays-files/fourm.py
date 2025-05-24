from typing import Any, Dict, List, Tuple, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from nanofm.modeling.transformer_layers import TransformerTrunk, TransformerDecoderTrunk, LayerNorm
from nanofm.utils.sampling import sample_tokens


#import os
#os.environ["TORCH_USE_CUDA_DSA"] = "1"


def build_1d_sincos_posemb(max_len, embed_dim=1024, temperature=10000.):
    """Sine-cosine positional embeddings from MoCo-v3, adapted back to 1d.
    Returns positional embedding of shape (N, D)
    """
    arange = torch.arange(max_len, dtype=torch.float32)  # Shape (N,)
    assert embed_dim % 2 == 0, 'Embed dimension must be divisible by 2 for 1D sin-cos position embedding'
    pos_dim = embed_dim // 2
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim  # Shape (D/2,)
    omega = 1. / (temperature ** omega)
    out = torch.einsum('n,d->nd', [arange, omega])  # Outer product, shape (N, D/2)
    pos_emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # Shape (N, D)
    return pos_emb


class FourM(nn.Module):
    """Simplified 4M definition, in which all modalities are handled in a single unified vocabulary.

    Args:
        enc_tokens_read_key: Key for reading encoder input tokens from the data dictionary
        dec_tokens_read_key: Key for reading decoder target tokens from the data dictionary.
            Only used for loss computation.
        enc_modalities_read_key: Key for reading encoder input modality ids from the data dictionary
        dec_modalities_read_key: Key for reading decoder target modality ids from the data dictionary
        enc_positions_read_key: Key for reading encoder input positions from the data dictionary
        dec_positions_read_key: Key for reading decoder target positions from the data dictionary
        enc_frame_positions_read_key: Key for reading encoder frame positions
        dec_frame_positions_read_key: Key for reading decoder frame positions
        enc_spatial_positions_read_key: Key for reading encoder spatial positions
        dec_spatial_positions_read_key: Key for reading decoder spatial positions
        enc_pad_mask_read_key: Key for reading encoder input padding mask from the data dictionary
        dec_pad_mask_read_key: Key for reading decoder target padding mask from the data dictionary
        modalities: List of modality names
        vocab_sizes: List of vocabulary sizes for each modality
        max_seq_lens: List of maximum sequence lengths for each modality
        dim: Transformer dimension
        enc_depth: Number of Transformer encoder layers
        dec_depth: Number of Transformer decoder layers
        head_dim: Dimension of each attention head
        mlp_ratio: Ratio of MLP hidden dimension to transformer dimension
        use_bias: Whether to include bias in the QKV, attention output projection and MLP layers
        padding_idx: Padding index for the target sequences
        init_std: Standard deviation for weight initialization
        per_modality_loss_avg: If True, compute the loss for each modality separately and average them.
            Otherwise, compute the loss over all target tokens together.
    """
    def __init__(
        self,
        enc_tokens_read_key: str,
        dec_tokens_read_key: str,
        enc_modalities_read_key: str,
        dec_modalities_read_key: str,
        enc_positions_read_key: str,
        dec_positions_read_key: str,
        enc_frame_positions_read_key: str,
        dec_frame_positions_read_key: str,
        enc_spatial_positions_read_key: str,
        dec_spatial_positions_read_key: str,
        enc_pad_mask_read_key: str,
        dec_pad_mask_read_key: str,
        modalities: List[str],
        vocab_sizes: List[int],
        max_seq_lens: List[int],
        dim: int = 512,
        enc_depth: int = 8,
        dec_depth: int = 8,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        use_bias: bool = False,
        padding_idx: int = -100,
        init_std: float = 0.02,
        per_modality_loss_avg: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.enc_tokens_read_key = enc_tokens_read_key
        self.dec_tokens_read_key = dec_tokens_read_key
        self.enc_modalities_read_key = enc_modalities_read_key
        self.dec_modalities_read_key = dec_modalities_read_key
        self.enc_positions_read_key = enc_positions_read_key
        self.dec_positions_read_key = dec_positions_read_key
        self.enc_frame_positions_read_key = enc_frame_positions_read_key
        self.dec_frame_positions_read_key = dec_frame_positions_read_key
        self.enc_spatial_positions_read_key = enc_spatial_positions_read_key
        self.dec_spatial_positions_read_key = dec_spatial_positions_read_key
        self.enc_pad_mask_read_key = enc_pad_mask_read_key
        self.dec_pad_mask_read_key = dec_pad_mask_read_key

        self.modalities = modalities
        self.vocab_sizes = vocab_sizes
        self.vocab_size = max(vocab_sizes)
        self.max_seq_lens = max_seq_lens
        self.max_posemb_len = max(max_seq_lens)
        self.num_modalities = len(modalities)
        self.padding_idx = padding_idx
        self.per_modality_loss_avg = per_modality_loss_avg

        # Initialize encoder token embedding
        self.enc_tok_emb = nn.Embedding(self.vocab_size, dim)

        # Initialize positional embeddings of predefined maximum length
        pos_emb = build_1d_sincos_posemb(self.max_posemb_len, dim)
        self.register_buffer("pos_emb", pos_emb)

        # Learned spatial (0..255 for 16x16 grid) and temporal (0..max_frames-1) embeddings
        self.spatial_pos_emb = nn.Embedding(16 * 16, dim)  # 16x16 grid for spatial positions
        self.frame_pos_emb = nn.Embedding(17, dim)  # Assuming max 17 frames

        # Initialize modality embeddings
        self.enc_mod_emb = nn.Embedding(self.num_modalities, dim)
        self.dec_mod_emb = nn.Embedding(self.num_modalities, dim)

        # Initialize Transformer encoder and decoder trunks
        self.encoder = TransformerTrunk(dim, enc_depth, head_dim, mlp_ratio, use_bias)
        self.decoder = TransformerDecoderTrunk(dim, dec_depth, head_dim, mlp_ratio, use_bias)

        # Initialize encoder -> decoder context projection
        self.dec_context_proj = nn.Linear(dim, dim, bias=use_bias)

        # Initialize decoder output projection
        self.to_logits = nn.Linear(dim, self.vocab_size, bias=False)

        # Initialize norm layers
        self.enc_norm = LayerNorm(dim, bias=use_bias)
        self.dec_norm = LayerNorm(dim, bias=use_bias)

        # Weight initialization
        self.init_std = init_std
        self.initialize_weights()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def initialize_weights(self) -> None:
        """Initialize the weights of the model."""
        self.apply(self._init_weights)
        nn.init.constant_(self.to_logits.weight, 0.1)
        
    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.init_std)

    def get_num_params(self, non_embedding=True) -> int:
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.enc_tok_emb.weight.numel()
            n_params -= self.enc_mod_emb.weight.numel()
            n_params -= self.dec_mod_emb.weight.numel()
            n_params -= self.to_logits.weight.numel()
            #n_params -= self.frame_pos_emb.weight.numel()
            #n_params -= self.spatial_pos_emb.weight.numel()  # Subtract spatial embedding params
        return n_params

    def forward_encoder(
        self,
        enc_input_tokens: torch.LongTensor,
        enc_input_modalities: torch.LongTensor,
        enc_input_positions: torch.LongTensor,
        enc_frame_positions: torch.LongTensor,
        enc_spatial_positions: torch.LongTensor,  # [B, N]
        enc_pad_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.
        """
        B, N = enc_input_tokens.shape

        # Embed input tokens
        x = self.enc_tok_emb(enc_input_tokens)  # Shape: [B, N, D]

        # Embed modality IDs and add to token embeddings
        x = x + self.enc_mod_emb(enc_input_modalities)  # Shape: [B, N, D]

        # Add positional embeddings
        enc_posembs = self.pos_emb[enc_input_positions]  # Shape: [B, N, D]
        x = x + enc_posembs

        # Add spatial positional embeddings
        x = x + self.spatial_pos_emb(enc_spatial_positions)  # Shape: [B, N, D]

        # Add frame positional embeddings
        x = x + self.frame_pos_emb(enc_frame_positions)  # Shape: [B, N, D]

        # Construct attention mask for padding
        enc_pad_attn_mask = repeat(enc_pad_mask, 'b n -> b m n', m=N) if enc_pad_mask is not None else None

        # Forward pass through the Transformer encoder
        x = self.encoder(x, enc_pad_attn_mask)

        # Apply encoder output normalization
        x = self.enc_norm(x)

        return x, enc_posembs

    def forward_decoder(
        self,
        dec_input_modalities: torch.LongTensor,
        dec_input_positions: torch.LongTensor,
        dec_frame_positions: torch.LongTensor,
        dec_spatial_positions: torch.LongTensor,  # [B, M]
        enc_context: torch.Tensor,
        enc_posembs: torch.Tensor,
        enc_pad_mask: Optional[torch.BoolTensor] = None,
        dec_pad_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the decoder.
        """
        B, M = dec_input_modalities.shape
        _, N, _ = enc_context.shape

        # Embed target modality IDs
        x = self.dec_mod_emb(dec_input_modalities)  # Shape: [B, M, D]

        # Add positional embeddings
        x = x + self.pos_emb[dec_input_positions]  # Shape: [B, M, D]

        # Add spatial positional embeddings
        x = x + self.spatial_pos_emb(dec_spatial_positions)  # Shape: [B, M, D]

        # Add frame positional embeddings
        x = x + self.frame_pos_emb(dec_frame_positions)  # Shape: [B, M, D]

        # Construct attention masks
        dec_pad_sa_mask = repeat(dec_pad_mask, 'b m -> b n m', n=M) if dec_pad_mask is not None else None
        dec_pad_xa_mask = repeat(enc_pad_mask, 'b n -> b m n', m=M) if enc_pad_mask is not None else None

        # Project encoder context
        context = self.dec_context_proj(enc_context)  # Shape: [B, N, D]
        context = context + enc_posembs  # Shape: [B, N, D]

        # Pass through the Transformer decoder
        x = self.decoder(x, context, sa_mask=dec_pad_sa_mask, xa_mask=dec_pad_xa_mask)  # Shape: [B, M, D]

        # Apply decoder output normalization
        x = self.dec_norm(x)  # Shape: [B, M, D]

        return x

    def forward_model(
        self,
        enc_input_tokens: torch.LongTensor,
        enc_input_modalities: torch.LongTensor,
        enc_input_positions: torch.LongTensor,
        enc_frame_positions: torch.LongTensor,
        enc_spatial_positions: torch.LongTensor,  # [B, N]
        dec_input_modalities: torch.LongTensor,
        dec_input_positions: torch.LongTensor,
        dec_frame_positions: torch.LongTensor,
        dec_spatial_positions: torch.LongTensor,  # [B, M]
        enc_pad_mask: Optional[torch.BoolTensor] = None,
        dec_pad_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the entire model.
        """
        # Encoder forward pass
        enc_x, enc_posembs = self.forward_encoder(
            enc_input_tokens,
            enc_input_modalities,
            enc_input_positions,
            enc_frame_positions,
            enc_spatial_positions,
            enc_pad_mask,
        )

        # Decoder forward pass
        dec_x = self.forward_decoder(
            dec_input_modalities,
            dec_input_positions,
            dec_frame_positions,
            dec_spatial_positions,
            enc_x,
            enc_posembs,
            enc_pad_mask,
            dec_pad_mask,
        )

        # Output logits
        logits = self.to_logits(dec_x)  # Shape: [B, M, vocab_size]
        return logits

    def compute_ce_loss(
        self,
        logits: torch.Tensor,
        target_seq: torch.LongTensor,
        padding_idx: int = -100,
        per_modality_loss_avg: bool = False,
        modality_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the cross-entropy loss given logits and target labels, ignoring padding tokens.
        """
        B, L, vocab_size = logits.shape
        logits = logits.reshape(-1, vocab_size)
        target_seq = target_seq.reshape(-1)

        if not per_modality_loss_avg:
            loss = F.cross_entropy(logits, target_seq, ignore_index=padding_idx)
            per_modality_losses = {}
        else:
            modality_indices = modality_indices.reshape(-1)
            valid_mask = target_seq != padding_idx

            losses = F.cross_entropy(logits, target_seq, reduction='none')

            per_modality_losses = {}
            for mod_idx, modality in enumerate(self.modalities):
                mod_mask = (modality_indices == mod_idx) & valid_mask
                if mod_mask.any():
                    per_modality_losses[modality] = losses[mod_mask].mean()
                else:
                    per_modality_losses[modality] = torch.tensor(0.0, device=losses.device)

            loss = sum(per_modality_losses.values()) / len(per_modality_losses)

        return loss, per_modality_losses

    def forward(self, data_dict: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through the model.
        """
        # Encoder, decoder, and output head forward pass
        logits = self.forward_model(
            enc_input_tokens=data_dict[self.enc_tokens_read_key],
            enc_input_modalities=data_dict[self.enc_modalities_read_key],
            enc_input_positions=data_dict[self.enc_positions_read_key],
            enc_frame_positions=data_dict[self.enc_frame_positions_read_key],
            enc_spatial_positions=data_dict[self.enc_spatial_positions_read_key],  # Added
            dec_input_modalities=data_dict[self.dec_modalities_read_key],
            dec_input_positions=data_dict[self.dec_positions_read_key],
            dec_frame_positions=data_dict[self.dec_frame_positions_read_key],
            dec_spatial_positions=data_dict[self.dec_spatial_positions_read_key],  # Added
            enc_pad_mask=data_dict.get(self.enc_pad_mask_read_key, None),
            dec_pad_mask=data_dict.get(self.dec_pad_mask_read_key, None),
        )

        # Compute loss
        loss, modality_metrics = self.compute_ce_loss(
            logits=logits,
            target_seq=data_dict[self.dec_tokens_read_key],
            padding_idx=self.padding_idx,
            per_modality_loss_avg=self.per_modality_loss_avg,
            modality_indices=data_dict[self.dec_modalities_read_key],
        )

        return loss, modality_metrics

    def get_unmasking_schedule(self, total_tokens: int, num_steps: int = 8) -> List[int]:
        """
        Generates a schedule for unmasking tokens at inference time.
        """
        assert total_tokens > 0, "No tokens to unmask in the input sequence."
        assert num_steps > 0, "Number of steps should be greater than zero."
        assert num_steps <= total_tokens, "Number of steps should be less than or equal to the total number of tokens to unmask."

        tokens_per_step = total_tokens // num_steps
        remainder = total_tokens % num_steps
        schedule = [tokens_per_step] * num_steps
        schedule[-1] += remainder

        assert len(schedule) == num_steps, "Schedule length should match the number of steps."
        assert sum(schedule) == total_tokens, "Total number of tokens to unmask should match the sum of the schedule."

        return schedule

    def generate_one_modality_roar(
        self,
        enc_input_tokens: torch.LongTensor,
        enc_input_positions: torch.LongTensor,
        enc_input_modalities: torch.LongTensor,
        enc_frame_positions: torch.LongTensor,
        enc_spatial_positions: torch.LongTensor,
        target_mod: str,
        num_steps: int = 8,
        temp: float = 1.0,
        top_p: float = 0.0,
        top_k: float = 0.0,
        ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """
            Generate one modality through iterative unmasking using the Random Order Auto Regressive (ROAR) decoding scheme.
        """
        B, N = enc_input_tokens.shape
        assert B == 1
        device = enc_input_tokens.device
        print(f"Device: {device}")
    
        target_mod_index = self.modalities.index(target_mod)
        n_tokens_target = self.max_seq_lens[target_mod_index]
        print(f"target_mod: {target_mod}, target_mod_index: {target_mod_index}, n_tokens_target: {n_tokens_target}")
    
        # Get schedule for unmasking tokens
        schedule = self.get_unmasking_schedule(n_tokens_target, num_steps)
        print(f"schedule: {schedule}")
    
        # Randomly shuffle positions from 0 to n_tokens_target - 1
        positions_order = np.random.permutation(n_tokens_target)
        indices = np.cumsum(schedule, axis=-1)[:-1]
        dec_input_positions_list = [torch.from_numpy(np.reshape(positions, (1, -1))) for positions in
                                    np.array_split(positions_order, indices, axis=-1)]
        print(f'dec_input_positions_list : {dec_input_positions_list}')
        # Initialize spatial and frame positions for the target modality
        print(device)
        max_frames = 17
        tokens_per_frame = 256
        dec_spatial_positions_list = []
        dec_frame_positions_list = []
        for step, k in enumerate(schedule):
            positions = dec_input_positions_list[step]
            print(positions)
            frame_indices = (positions // tokens_per_frame)
            spatial_indices = (positions % tokens_per_frame)
            dec_frame_positions_list.append(frame_indices)
            dec_spatial_positions_list.append(spatial_indices)
        enc_frame_positions   = enc_frame_positions.unsqueeze(0)
        enc_spatial_positions = enc_spatial_positions.unsqueeze(0)
        for step, k in enumerate(schedule):
            dec_input_positions = dec_input_positions_list[step]
            dec_frame_positions = dec_frame_positions_list[step]
            dec_spatial_positions = dec_spatial_positions_list[step]
    
            dec_input_modalities = target_mod_index * torch.ones(1, k, dtype=torch.long )
            print(f"dec_input_modalities shape: {dec_input_modalities.shape}, device: {dec_input_modalities.device}")
    
            # Compute padding masks
            enc_pad_mask = enc_input_tokens != self.padding_idx  # Use self.padding_idx directly
            dec_pad_mask = torch.ones_like(dec_input_modalities,  dtype=torch.bool)
            
            # print(enc_input_tokens.device)
            # print(enc_input_modalities.device)
            # print(enc_input_positions.device)
            # print(enc_frame_positions.device)
            # print(enc_spatial_positions.device)
            # print(dec_input_modalities.device)
            # print(dec_input_positions.device)
            # print(dec_frame_positions.device)
            # print(dec_spatial_positions.device)
            # print(enc_pad_mask.device)
            # print(dec_pad_mask.device)
            print(f'the enc tokens: {enc_input_tokens}')
            print(f'the enc modalities: {enc_input_modalities}')
            print(f'the enc positions: {enc_input_positions}')
            print(f'the enc frame positions: {enc_frame_positions}')
            print(f'the enc  spatial positions: {enc_spatial_positions}')
            print(f'the decoder modalities: {dec_input_modalities.to(device)}')
            print(f'the decoder positions: {dec_input_positions.to(device)}')
            print(f'the decoder frame positions: {dec_frame_positions.to(device)}')
            print(f'the decoder spatial positions: {dec_spatial_positions.to(device)}')
            print(f'the enc padding mask: {enc_pad_mask}')
            print(f'the decoder padding mask: {dec_pad_mask.to(device)}')
            
            # Forward pass through the model
            predicted_logits = self.forward_model(
                enc_input_tokens,
                enc_input_modalities,
                enc_input_positions,
                enc_frame_positions,
                enc_spatial_positions,
                dec_input_modalities.to(device),
                dec_input_positions.to(device),
                dec_frame_positions.to(device),
                dec_spatial_positions.to(device),
                enc_pad_mask=enc_pad_mask,
                dec_pad_mask=dec_pad_mask.to(device),
            )[0]
            
            # Sample new tokens
            samples, _ = sample_tokens(predicted_logits, temp, top_k, top_p)
    
            # Concatenate new tokens to encoder inputs
            enc_input_tokens = torch.cat([enc_input_tokens, samples.unsqueeze(0)], dim=1)
            enc_input_positions = torch.cat([enc_input_positions, dec_input_positions.to(device)], dim=1)
            enc_input_modalities = torch.cat([enc_input_modalities, dec_input_modalities.to(device)], dim=1)
            enc_frame_positions = torch.cat([enc_frame_positions, dec_frame_positions.to(device)], dim=1)
            enc_spatial_positions = torch.cat([enc_spatial_positions, dec_spatial_positions.to(device)], dim=1)
    
        # Select and unshuffle predicted tokens
        print(enc_input_tokens.squeeze(0).shape)
        print(enc_input_modalities.squeeze(0).shape)
        print(enc_input_modalities == target_mod_index)
        pred_tokens = enc_input_tokens.squeeze(0)[ (enc_input_modalities == target_mod_index)[0]]
        indices = enc_input_positions.squeeze(0)[ (enc_input_modalities == target_mod_index)[0]]
        pred_tokens = pred_tokens[indices.argsort()].unsqueeze(0)
    
        return pred_tokens, enc_input_tokens, enc_input_positions, enc_input_modalities, enc_frame_positions, enc_spatial_positions