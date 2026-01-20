# ToDo:

# 1. Add source title also
# 2. Add transliterations of verse and ref
# 3. Link next and previous verses

# 4. Regular testing on val set


from basic import TasnsfModel
from tokenizer.tokenizer import encode
from model_config import model_config

import torch
import torch.nn as nn
import numpy as np
import json
import math
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset


device = "cuda:3"
save_path = "../models"
data_path = "../../data/"

max_seq_len = model_config["max_seq_len"]
batch_size = 8
model_save_step = 10000
plot_save_step = 100
loss_upd_step = 10

pad_id = encode("<pad>")[0]

base_lr = 1e-4
eta_min = base_lr * 0.1
total_steps = 150000
warmup_steps = 1500



def lr_lambda(current_step):
    if current_step < warmup_steps:
        return current_step / max(1, warmup_steps)

    if current_step > total_steps:
        return 0.1
            
    progress = (current_step - warmup_steps) / max(1, (total_steps - warmup_steps))
    return max((eta_min / base_lr), 0.5 * (1.0 + math.cos(progress * math.pi)))


class DharaDataset(Dataset):
    def __init__(self, verse_prob=0.4, word_prob=0.3, chunk_prob=0.3):
        
        self.verse_prob = verse_prob
        self.word_prob = word_prob
        self.chunk_prob = chunk_prob
        
        self.verse_tokens = np.memmap("../../data/verse_train_tokens.bin", dtype=np.uint16, mode="r")
        self.word_tokens = np.memmap("../../data/word_train_tokens.bin", dtype=np.uint16, mode="r")
        self.chunk_tokens = np.memmap("../../data/chunk_train_tokens.bin", dtype=np.uint16, mode="r")
        
        
    def __len__(self):
        return 10**9
    
    
    def _sample_window(self, tokens):
        start = np.random.randint(0, len(tokens) - max_seq_len - 1)
        x = tokens[start : start + max_seq_len]
        return torch.tensor(x, dtype=torch.long)
    
    
    def __getitem__(self, index):

        r = np.random.rand()
        
        if r < self.verse_prob:
            return self._sample_window(self.verse_tokens)
        elif r < self.verse_prob + self.word_prob:
            return self._sample_window(self.word_tokens)
        else:
            return self._sample_window(self.chunk_tokens)
        
    
    
def collate_fn(batch):
    return torch.stack(batch)


dataset = DharaDataset(verse_prob=0.4, word_prob=0.3, chunk_prob=0.3)
loader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2)


model = TasnsfModel(**model_config).to(device)
loss_fn = nn.CrossEntropyLoss()  # ignore_index=pad_id
optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

print("Model Parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
# 253,308,928

avg_loss = 10.2
decay_rate = 0.01
losses = []
model.train()

for step, x in enumerate(loader):
    x = x.to(device)
    logits = model(x)
    
    targets = x[:, 1:]
    logits = logits[:, :-1, :]
    
    targets = targets.reshape(-1)
    logits = logits.reshape(-1, logits.shape[-1])
    
    loss = loss_fn(logits, targets)
    curr_loss = loss.item()
    
    loss.backward()
    optimizer.step()
    
    scheduler.step()
    optimizer.zero_grad()
    
    avg_loss = (1 - decay_rate) * avg_loss + decay_rate * curr_loss
    if step % loss_upd_step == 0: 
        losses.append(avg_loss)
        print(f"step: {step}   loss: {round(curr_loss, 3)}   avg: {round(avg_loss, 3)}   lr: {optimizer.param_groups[0]["lr"]}")

    if step % model_save_step == 0:
        torch.save(model.state_dict(), f"{save_path}/step_{step}.pth")
        
    if step % plot_save_step == 0:            
        plt.figure(figsize=(16,8))
        plt.plot(losses)
        plt.grid(True)
        plt.savefig(f"{save_path}/loss_curve.png")
        plt.close()

        
