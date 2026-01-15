# IKS GPT - Building a Transformer-Based Language Model (300M) for Indian Knowledge Systems

*This is an extension of the [scratch_former](https://github.com/rahul3613/scratch_former) project for Indian Knowledge Systems*


A GPT-like transformer model built from scratch for generating and understanding text from Indian Knowledge Systems (IKS), including Sanskrit verses, definitions, and historical texts. This project implements a complete pipeline from tokenization to model training and inference.

## Project Overview

This repository contains a comprehensive implementation of a transformer-based language model designed to work with Indian Knowledge Systems data. The project includes:

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

- **`model_config.py`**: Hyperparameter configuration for the **300M** parameter transformer model
  - Vocabulary size: 25,000 tokens
  - Sequence length: 1,024 tokens
  - Embedding dimension: 1,024
  - Number of attention heads: 16
  - Number of transformer layers: 15

### Training Scripts

- **`train.py`**: Standard PyTorch training implementation
  - Trains on mixed data sources (verses, words, chunks) with configurable probabilities
  - Implements cosine annealing learning rate schedule with warmup
  - Model checkpoints saved in `models/` directory

- **`ft_train.py`**: Fine-tuning training script
  - Loads pre-trained model from `models/epoch_1.pth`
  - Trains on question-answer type data
  - Saves fine-tuned models in `ft_models/` directory

- **`pl_train.py`**: PyTorch Lightning training implementation
  - Distributed training support with PyTorch Lightning
  - Checkpoint management and logging using Lightning callbacks
  - Logs saved in `pl_models/lightning_logs/` directory

### Testing & Inference

- **`pl_test.py`**: Text generation script
  - Loads checkpoint from PyTorch Lightning training
  - Implements inference with temperature, Top-p (nucleus) and Repetition penalty
  - Example: Generates continuations from seed text like "sita the daughter of"

- **`test_script.ipynb`**, **`ft_test.ipynb`**, **`pl_test.ipynb`**, **`test.ipynb`**: Jupyter notebooks for testing and experimentation

### Tokenization & Utilities

- **`tokenizer/tokenizer.py`**: BPE tokenizer implementation
  - Uses pre-learned merges from JSON files
  - Encodes text to token IDs and Decodes token IDs back to text
  - Supports special tokens (`<pad>`, `<eos>`, `<system>`)

- **`tokenizer/bpe.py`**: BPE (Byte Pair Encoding) tokenizer training
  - Builds vocabulary from verse, word, and chunk data
  - Learns merge operations for frequent byte-pair combinations
  - Outputs vocabulary with frequency information

- **`tokenizer/merges.json`**, **`tokenizer/merges_spl.json`**: Pre-learned BPE merge vocabularies

- **`utils.py`**: Training utility functions
  - `aks_translit()`: Script transliteration support
  - `shuff_drop()`: Text shuffling and dropping augmentation
  - `add_noise()`: Character-level noise addition (shuffling, replacement, displacement)

- **`ft_utils.py`**: Fine-tuning utility functions
  - `add_noise()`: Enhanced noise augmentation with stop-word aware dropping
  - `get_train_ids()`: Data loader for verse, word, and chunk training data
  - Stop words list for intelligent text augmentation

### Saved Models & Logs

- **`models/`**: Stores checkpoints from standard training
  - `step_{n}.pth`: Model checkpoint after `n` steps

- **`ft_models/`**: Stores checkpoints from fine-tuning
  - Contains fine-tuned model states

- **`pl_models/`**: PyTorch Lightning training artifacts
  - `checkpoints/`: Model checkpoints
  - `lightning_logs/`: *TensorBoard logs* and *hparams*

## Model Architecture

### Transformer Block
Each transformer block consists of:
1. **Multi-Head Self-Attention**
   - 16 parallel attention heads
   - Query, Key, Value projections
   - Scaled dot-product attention
   - Dropout for regularization

2. **Feed-Forward Network**
   - Gated linear unit (GLU) activation: `SiLU(gate) * up_proj`
   - 4x expansion followed by projection back to embedding dimension
   - Dropout for regularization

3. **Residual Connections & Layer Normalization**
   - Pre-normalization architecture
   - Layer norm before both attention and feed-forward
   - Residual skip connections

### Full Model
- Token embedding + positional embedding
- 15 stacked transformer blocks
- Output projection to vocabulary size
- Causal masking for autoregressive generation

## Training Details

### Datasets
- **Verses**: Sanskrit verses from Indian Knowledge Systems
- **Words**: Dictionary words with definitions
- **Chunks**: Larger text passages and chapters

### Data Augmentation
- Random character shuffling
- Word dropping with stop-word awareness
- Character replacement and displacement
- Space/newline manipulation

### Learning Rate Schedule
- Linear warmup for first 1,500 steps
- Cosine annealing with 150,000 total training steps
- Minimum learning rate: 10% of base learning rate

## Usage Examples

### Training
```bash
# Standard training
python train.py

# Fine-tuning pre-trained model
python ft_train.py

# PyTorch Lightning training
python pl_train.py
```

### Inference
```bash
# Generate text from seed
python pl_test.py
```

### Tokenization
```python
from tokenizer.tokenizer import encode, decode

# Tokenize text
tokens = encode("sita the daughter of")

# Decode tokens back to text
text = decode(tokens)
```

## Dependencies

Key libraries used:
- `torch`: PyTorch deep learning framework
- `pytorch-lightning`: Training framework for distributed/multi-GPU training
- `aksharamukha`: Script transliteration for Indian languages
- `numpy`: Array operations
- `regex`: Enhanced regex for BPE tokenization
- `tqdm`: Progress bars
- `matplotlib`: Plotting training curves

## Key Features

- **Autoregressive Generation**: Predicts next token given context
- **Causal Masking**: Prevents attention to future tokens during training
- **Efficient Data Loading**: Memory-mapped data files for large datasets
- **Multi-source Training**: Simultaneously trains on verses, words, and chunks
- **Checkpoint Management**: Automatic saving and resuming of training
- **Fine-tuning Support**: Pre-trained model loading for downstream tasks
- **Multiple Training Backends**: Standard PyTorch, distributed Lightning, and fine-tuning implementations

## Output Examples

The model can generate continuations of prompts like:
```
Seed: "sita the daughter of"
Generated: "[model continues with relevant text]"
```

With configurable generation parameters:
- **Temperature**: Controls randomness (higher = more random)
- **Top-p**: Nucleus sampling for diversity
- **Repetition Penalty**: Discourages token repetition

## Notes

- Pre-trained models are loaded from `models/` and `pl_models/checkpoints/` directories
- Data is expected in parent directory structure (`../data/`) with verse, word, and chunk subdirectories
- All training uses CUDA GPUs with device selection via `cuda:X` parameters
- Experiments are tracked in lightning logs for monitoring training progress
