## IKS GPT - Basic Version using  PyTorch only and single GPU training

### Training Scripts

- **`train.py`**: Standard PyTorch training implementation
  - Trains on mixed data sources (verses, words, chunks) with configurable probabilities
  - Implements cosine annealing learning rate schedule with warmup
  - Model checkpoints saved in `models/` directory

- **`ft_train.py`**: Fine-tuning training script
  - Loads pre-trained model from `models/`
  - Trains on question-answer type data
  - Saves fine-tuned models in `ft_models/` directory

### Testing & Inference

Scripts for loading the checkpoints for inference and testing. With options for setting parameters like temperature, Top-p (nucleus) and Repetition penalty
- **`test.ipynb`**: Jupyter notebook for using the model trained using the standard PyTorch script.
- **`ft_test.ipynb`**: Jupyter notebook for testing the answer generation for questions using the finetuned models.

- **`utils.py`**, **`ft_utils.py`** : Utility functions for training and finetuning the model with functions like `aks_translit()`, `shuff_drop()`, and `add_noise()`


## Usage Examples

### Training
```bash
# Standard training
python train.py

# Fine-tuning pre-trained model
python ft_train.py
```

### Inference
```bash
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