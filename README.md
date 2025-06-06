# 4M Foundation Model

## Overview

This repository contains the files of our project and extensions in Foundation Of Models based on a light version of the **4M** model. The pipeline is organized into four main stages—each with its own top-level directory:

```
├── preprocessing/
├── tokenization/
├── tokenizer/
└── generation/
```


## Architecture

1. **preprocessing/**  
   - Extract raw data (e.g., video → frames, audio clips, text transcripts)  
   - Ensure uniform inputs (e.g., a fixed number of frames per video)

2. **tokenization/**  
   - Experiment and finalize how each modality gets split into tokens 
   - Provide Jupyter notebooks that document and run these tokenization strategies

3. **tokenizer/**  
   - Train and inference files for the tokenizers (Audio, Image, Text)  
   - Produce discrete token IDs for all inputs, saved to disk for downstream use

4. **generation/**  
   - Define the multimodal transformer model nano4M that takes audio, image, and text tokens  
   - Include training scripts (`run_training.py`), configuration files, and inference notebooks for generation

---

## IMPORTANT REMARKS 
The following files are not included in this repository, but are essential for the project and are contained on SCITAS 
at the correspoding path:

- `tokenizer/Image/checkpoint_image/vqgan_f16_epoch1.pth` : is contained in the SCITAS path `/work/com-304/IAY_neurons_u2/Tokenizers/Image/checkpoint_image/vqgan_f16_epoch1.pth`
- `tokenizer/Text/checkpoint_text_tokenizer/checkpoint-31000/model.safetensors` : is contained in the SCITAS path `/work/com-304/IAY_neurons_u2/Tokenizers/Text/checkpoint_text_tokenizer/checkpoint-31000/model.safetensors`
- `tokenizer/Text/checkpoint_text_tokenizer/checkpoint-31000/optimizer.pt` : is contained in the SCITAS path `/work/com-304/IAY_neurons_u2/Tokenizers/Text/checkpoint_text_tokenizer/checkpoint-31000/optimizer.pt`
- `tokenizer/Text/checkpoint_text_tokenizer/checkpoint-31000/model.safetensors` : is contained in the SCITAS path `/work/com-304/IAY_neurons_u2/Tokenizers/Text/checkpoint_text_tokenizer/checkpoint-31000/model.safetensors`
