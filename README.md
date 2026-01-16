# IKS GPT - Building a Transformer-Based Language Model (300M) for Indian Knowledge Systems

*This is an extension of the [scratch_former](https://github.com/rahul3613/scratch_former) project for Indian Knowledge Systems*


A GPT-like transformer model built from scratch for generating and understanding text from Indian Knowledge Systems (IKS), including Sanskrit verses, definitions, and historical texts. This project implements a complete pipeline from tokenization to model training and inference.

## Project Overview

- **Custom Transformer Architecture**: From-scratch implementation of multi-head attention, feed-forward networks, and transformer blocks
- **Tokenization**: BPE-based efficient tokenization for IKS text with support for special tokens and multiple scripts
- **Training Pipelines**: Multiple training approaches including standard PyTorch, fine-tuning, and PyTorch Lightning implementations
- **Data Processing**: Utilities for handling verses, words, and text chunks with noise augmentation
- **Inference**: Text generation with sampling strategies (top-p filtering, repetition penalty)

## Repository Structure

### Core Model Files

- **`basic.py`**: Core transformer architecture implementation
  - `SelfAttn`: Single-head self-attention mechanism
  - `MultiHeadAttn`: Multi-head attention with multiple parallel attention heads
  - `TransfBlock`: Transformer block combining attention and feed-forward layers
  - `TasnsfModel`: Complete transformer model with embeddings and output projection

- **`model_config.py`**: Hyperparameter configuration for a **300M** parameter transformer model
  - Vocabulary size: 25,000
  - Sequence length: 1,024
  - Embedding dimension: 1,024
  - Number of attention heads: 16
  - Number of transformer layers: 15

### Training Scripts

- **`train.py`**: Standard PyTorch training implementation
  - Trains on mixed data sources (verses, words, chunks) with configurable probabilities
  - Implements cosine annealing learning rate schedule with warmup
  - Model checkpoints saved in `models/` directory

- **`pl_train.py`**: PyTorch Lightning training implementation
  - Distributed training support with PyTorch Lightning
  - Checkpoint management and logging using Lightning callbacks
  - Logs saved in `pl_models/lightning_logs/` directory

- **`ft_train.py`**: Fine-tuning training script
  - Loads pre-trained model from `models/`
  - Trains on question-answer type data
  - Saves fine-tuned models in `ft_models/` directory

### Testing & Inference

Scripts for loading the checkpoints for inference and testing. With options for setting parameters like temperature, Top-p (nucleus) and Repetition penalty
- **`pl_test.py`**: Text generation script for the model trained using PyTorch Lightning.
- **`test.ipynb`**: Jupyter notebook for using the model trained using the standard PyTorch script.
- **`ft_test.ipynb`**: Jupyter notebook for testing the answer generation for questions using the finetuned models.

### Tokenization & Utilities

- **`tokenizer/bpe.py`**: BPE (Byte Pair Encoding) tokenizer training
  - Builds vocabulary from verse, word, and chunk data
  - Learns merge operations for frequent byte-pair combinations
  - Outputs vocabulary with frequency information

- **`tokenizer/merges.json`**, **`tokenizer/merges_spl.json`**: Learned BPE merge vocabularies

- **`tokenizer/tokenizer.py`**: BPE tokenizer implementation
  - Uses pre-learned merges from JSON files
  - Encodes text to token IDs and Decodes token IDs back to text
  - Supports special tokens (`<pad>`, `<eos>`, `<user>`, `<system>`)
  - Usage:
    ```python
    from tokenizer.tokenizer import encode, decode

    # Tokenize text
    tokens = encode("sita the daughter of")

    # Decode tokens back to text
    text = decode(tokens)
    ```

- **`utils.py`**, **`ft_utils.py`** : Utility functions for training and finetuning the model with functions like `aks_translit()`, `shuff_drop()`, and `add_noise()`


## Datasets
- **Verses**: Sanskrit verses from Indian Knowledge Systems
- **Words**: Dictionary words with definitions
- **Chunks**: Larger text passages and chapters


## Usage Examples

### Training
```bash
# Standard training
python train.py

# PyTorch Lightning training
python pl_train.py

# Fine-tuning pre-trained model
python ft_train.py
```

### Inference
```bash
# Generate text from seed
python pl_test.py

test.ipynb

# Notebook for question answering
ft_test.ipynb
```
### Sample Question Answering
```python
question = '''meaning of "dharma"?'''
prompt = "<user>" + question + "<system>"

print(generate(prompt))

# <user>meaning of "dharma"?<system>
# the term "dharma" refers to a specific duty or office that is considered righteous. it can refer to the duties of a king, a king, and also to the duties of an ascetic (stayed in pious practices).<eos>
```