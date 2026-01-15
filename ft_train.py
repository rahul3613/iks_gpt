from basic import TasnsfModel
from tokenizer.tokenizer import encode
from ft_utils import get_train_ids, get_verse_data, get_chunk_data, get_word_data
from model_config import model_config

import torch
import torch.nn as nn
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset


device = "cuda:2"
save_path = "ft_models"
pt_model_path = "models/epoch_1_step_150000.pth"

max_seq_len = model_config["max_seq_len"]
batch_size = 10
model_save_step = 10000
plot_save_step = 100
loss_upd_step = 10

pad_id = encode("<eos>")[0]
sys_id = encode("<system>")[0]

warmup_steps = int(15000 / batch_size)
lr = 5e-6


def lr_lambda(current_step):
    if current_step < warmup_steps:
        return current_step / max(1, warmup_steps)

    return 1.0


class VerseDataset(Dataset):
    def __init__(self):        
        self.all_ids = get_train_ids()
        
        
    def __len__(self):
        return len(self.all_ids)
    
    
    def __getitem__(self, index):

        id = self.all_ids[index]
        
        if id.startswith("v"):
            text = get_verse_data(id)
        
        if id.startswith("w"):
            text = get_word_data(id)
        
        if id.startswith("c"):
            text = get_chunk_data(id)
        
        tokens = encode(text)
        return torch.tensor(tokens, dtype=torch.long)
    
    
def collate_fn(batch):
    lengths = [len(b) for b in batch]
    max_len = min(max(lengths), max_seq_len)
    
    padded = []
    for b in batch:
        if len(b) < max_len:
            b = torch.cat([b, torch.full((max_len - len(b),), pad_id)])
        else:
            b = b[:max_len]
        padded.append(b)
            
    return torch.stack(padded)


dataset = VerseDataset()
loader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True, persistent_workers=True, prefetch_factor=10)


model = TasnsfModel(**model_config).to(device)
model.load_state_dict(torch.load(pt_model_path, map_location=device))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


epoch = 0
avg_loss = 2.5
decay_rate = 0.002
losses = []
model.train()
while True:
    epoch += 1
    print("epoch: ", epoch)

    for step, x in enumerate(loader):
        x = x.to(device)
        logits = model(x)
        
        targets = x.clone()
        for i in range(x.size(0)):
            sys_pos = (x[i] == sys_id).nonzero(as_tuple=True)[0]
            if len(sys_pos) > 0:
                ans_start = sys_pos[0] + 1
                targets[i, :ans_start] = -100
            else:
                targets[i, :] = -100
        
        targets = targets[:, 1:]
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
            print(f"step: {epoch}/{step}   loss: {round(curr_loss, 3)}   avg: {round(avg_loss, 3)}   lr: {optimizer.param_groups[0]["lr"]}")
        
        if step > 0:
            if step % model_save_step == 0:
                torch.save(model.state_dict(), f"{save_path}/epoch_{epoch}_step_{step}.pth")
                
            if step % plot_save_step == 0:            
                plt.figure(figsize=(16,8))
                plt.plot(losses)
                plt.grid(True)
                plt.savefig(f"{save_path}/loss_curve.png")
                plt.close()

    torch.save(model.state_dict(), f"{save_path}/epoch_{epoch}.pth")
        
