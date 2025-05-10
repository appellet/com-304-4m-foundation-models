# Copyright 2025 EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Callable, Optional
import os
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class AdaptedMultimodalDataset(Dataset):
    def __init__(
            self, 
            root_dir: str,
            modalities: List[str],
            transforms: Optional[Callable] = None,
            text_tokenizer_path: str = 'gpt2',
            text_max_length: int = 256,
        ):
        """
        Adapted multimodal dataset for the nano4m project.

        Assumptions:
        - Data is stored in {root_dir}/{sample_name}/{file_name}.{ext}.
        - Each sample directory contains rgb_tokens.npy, keypoints.npy, and caption.json.
        - There are no splits (e.g., train/val/test) or modality-named subdirectories.
        - Each modality has a single set of tokens (no augmentations to sample from).

        Args:
            root_dir: Root directory of the dataset (e.g., /work/com-304/IAY_neurons_u2/dataset/tokenized).
            modalities: List of modalities (e.g., ["rgb_tokens", "keypoints", "caption"]).
            transforms: Transformations to apply to the data_dict before returning.
            text_tokenizer_path: Path or HuggingFace Hub ID of the text tokenizer, e.g., gpt2.
            text_max_length: Maximum length of the text.
        """
        self.root_dir = root_dir
        self.modalities = modalities
        self.transforms = transforms
        
        self.file_names = self._get_file_names()

        self.text_tokenizer_path = text_tokenizer_path
        self.text_max_length = text_max_length
        self.text_tokenizer = self._get_text_tokenizer()
        
    def __len__(self):
        return len(self.file_names)

    def _get_file_names(self):
        # Get all sample directories in the root_dir
        sample_dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        return sorted(sample_dirs)
    
    def _get_text_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.text_tokenizer_path)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.add_special_tokens({
            'bos_token': '[SOS]',
            'eos_token': '[EOS]',
        })
        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single="[SOS] $A [EOS]",
            special_tokens=[('[EOS]', tokenizer.eos_token_id), ('[SOS]', tokenizer.bos_token_id)],
        )
        return tokenizer
        
    def __getitem__(self, idx):
        sample_name = self.file_names[idx]
        sample_dir = os.path.join(self.root_dir, sample_name)
        
        data_dict = {}

        for modality in self.modalities:
            if modality == "rgb_tokens":
                file_path = os.path.join(sample_dir, "rgb_tokens.npy")
                tokens_array = np.load(file_path)
                tokens = torch.from_numpy(tokens_array).long()
            elif modality == "keypoints":
                file_path = os.path.join(sample_dir, "keypoints.npy")
                tokens_array = np.load(file_path)
                tokens = torch.from_numpy(tokens_array).long()
            elif modality == "caption":
                file_path = os.path.join(sample_dir, "caption.json")
                with open(file_path, 'r') as f:
                    captions = json.load(f)
                # Since there's no augmentation sampling, take the first caption (or only caption)
                caption = captions[0] if isinstance(captions, list) else captions
                tokenized = self.text_tokenizer(
                    caption, max_length=self.text_max_length, padding='max_length', 
                    truncation=True, return_tensors='pt'
                )
                tokens = tokenized['input_ids'][0]
            else:
                raise ValueError(f"Unknown modality: {modality}")

            data_dict[modality] = tokens

        if self.transforms:
            data_dict = self.transforms(data_dict)
        
        return data_dict