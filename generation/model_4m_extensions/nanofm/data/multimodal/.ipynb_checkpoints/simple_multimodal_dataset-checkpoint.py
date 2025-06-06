# simple_multimodal_dataset.py

import os
from pathlib import Path
from typing import List, Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SimpleMultimodalDataset(Dataset):
    """
    A simple multimodal dataset loader for 3 modalities: 'rgb', 'pose', and 'caption'.
    For 'rgb' and 'pose', each sample is a .npz with a [17,16,16] 'tokens' array.
    We flatten those into a 1D sequence of length 17*16*16=4352, and generate:
      - spatial_positions: 0..255 repeated for each of the 17 frames
      - frame_positions: 0..16 each repeated 256 times
    For 'caption', we tokenize text to a fixed length and use zeros for spatial/frame positions.
    """
    def __init__(
        self,
        root_dir: str,
        split: str,
        modalities: List[str],
        transforms: Optional[Callable] = None,
        sample_from_k_augmentations: int = 10,
        text_tokenizer_path: str = 'gpt2',
        text_max_length: int = 256,
    ):
        self.root_dir = root_dir
        self.split = split
        self.modalities = modalities
        self.transforms = transforms
        self.sample_from_k_augmentations = sample_from_k_augmentations

        # collect all file stems from the first modality directory
        first_mod_dir = Path(root_dir) / split / modalities[0]
        self.file_names = sorted(p.stem for p in first_mod_dir.iterdir())

        # detect file extensions for each modality
        self.modality_extensions = {
            mod: next((Path(root_dir) / split / mod).glob('*')).suffix
            for mod in modalities
        }

        # set up text tokenizer
        self.text_max_length = text_max_length
        self.text_tokenizer = self._get_text_tokenizer(text_tokenizer_path)

    def __len__(self):
        return len(self.file_names)

    def _get_text_tokenizer(self, model_name: str):
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.add_special_tokens({'pad_token': '[PAD]'})
        tok.add_special_tokens({'bos_token': '[SOS]', 'eos_token': '[EOS]'})
        tok._tokenizer.post_processor = TemplateProcessing(
            single='[SOS] $A [EOS]',
            special_tokens=[('[SOS]', tok.bos_token_id), ('[EOS]', tok.eos_token_id)]
        )
        return tok

    def __getitem__(self, idx: int):
        fname = self.file_names[idx]
        data_dict = {}

        for mod in self.modalities:
            ext = self.modality_extensions[mod]
            path = Path(self.root_dir) / self.split / mod / f"{fname}{ext}"

            if mod in ['rgb', 'pose']:
                # load .npz containing tokens of shape [17,16,16]
                arr = np.load(path)
                tokens_np = arr['tokens']              # shape: [n_frames, H, W]
                n_frames, H, W = tokens_np.shape        # expect 17,16,16

                # flatten to [n_frames*H*W]
                tokens = torch.from_numpy(tokens_np).long().view(-1)  

                # spatial IDs: 0..(H*W-1) repeated for each frame
                spatial_positions = torch.arange(H * W, dtype=torch.long).repeat(n_frames)

                # frame IDs: each frame index repeated H*W times
                frame_positions   = torch.arange(n_frames, dtype=torch.long).repeat_interleave(H * W)

                data_dict[f"{mod}_tokens"]            = tokens
                data_dict[f"{mod}_spatial_positions"] = spatial_positions
                data_dict[f"{mod}_frame_positions"]   = frame_positions

            elif mod == 'caption':
                # load and tokenize text
                txt = path.read_text().strip()
                toks = self.text_tokenizer(
                    txt,
                    max_length=self.text_max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )['input_ids'][0]  # shape: [text_max_length]

                # no spatial/frame for text: use zeros
                zeros = torch.zeros_like(toks, dtype=torch.long)
                data_dict['caption_tokens']            = toks
                data_dict['caption_spatial_positions'] = zeros
                data_dict['caption_frame_positions']   = zeros

            else:
                raise ValueError(f"Unknown modality: {mod}")

        if self.transforms:
            data_dict = self.transforms(data_dict)

        return data_dict