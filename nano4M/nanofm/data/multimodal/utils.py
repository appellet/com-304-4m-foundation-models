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

from typing import List, Dict, Any
import copy


def to_unified_multimodal_vocab(data_dict: Dict[str, Any], modalities: List[str], vocab_sizes: List[int]):
    """
    Shifts the token values for each modality so that each modality's tokens fall in a unique range
    defined by the cumulative vocabulary sizes, useful for training a multimodal model with a unified
    embedding space.

    For example, if modalities = ['rgb_tokens', 'keypoints', 'caption']
    and vocab_sizes = [16384, 16384, 50257], then the tokens for:
      - 'rgb_tokens' remain in [0, 16383],
      - 'keypoints' will be shifted to [16384, 32767],
      - 'caption' will be shifted to [32768, 83024].

    Args:
        data_dict: Input dictionary mapping modalities (e.g., 'rgb_tokens', 'keypoints', 'caption') to token tensors.
        modalities: List of modality names specifying the order of allocation (e.g., ['rgb_tokens', 'keypoints', 'caption']).
        vocab_sizes: List of vocabulary sizes for each modality (e.g., [16384, 16384, 50257]).

    Returns:
        A transformed copy of data_dict where the tokens for each modality are shifted to the unified vocabulary.
    """
    new_data_dict = copy.deepcopy(data_dict)
    
    offset = 0
    for modality, vocab_size in zip(modalities, vocab_sizes):
        if modality in new_data_dict:
            # Shift token indices by the current cumulative offset
            new_data_dict[modality] = new_data_dict[modality] + offset
        offset += vocab_size  # Increment offset for the next modality
    return new_data_dict

def from_unified_multimodal_vocab(data_dict: Dict[str, Any], modalities: List[str], vocab_sizes: List[int]):
    """
    Inverse function of to_unified_multimodal_vocab. Subtracts the previously added offsets from each
    modality's token values to restore the original token indices.

    Args:
        data_dict: The transformed dictionary, with shifted token values.
        modalities: The modalities in the same order used for the forward transform (e.g., ['rgb_tokens', 'keypoints', 'caption']).
        vocab_sizes: The vocabulary sizes for each modality in the same order (e.g., [16384, 16384, 50257]).

    Returns:
        A new dictionary with the original token values restored.
    """
    new_data_dict = copy.deepcopy(data_dict)
    
    offset = 0
    for modality, vocab_size in zip(modalities, vocab_sizes):
        if modality in new_data_dict:
            # Subtract the cumulative offset to recover original tokens
            new_data_dict[modality] = new_data_dict[modality] - offset
        offset += vocab_size
    return new_data_dict