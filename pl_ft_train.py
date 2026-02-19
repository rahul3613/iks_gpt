from basic import TasnsfModel
from model_config import model_config

import torch
import torch.nn as nn
import numpy as np
import math
import random
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# Val
from tokenizer.tokenizer import encode
# from val_utils import get_eval_ids, get_verse_data, get_chunk_data, get_word_data
from ft_utils import get_train_ids, get_val_ids, get_verse_data, get_chunk_data, get_word_data

ckpt_path = "pl_models/checkpoints_v4/step=90000.ckpt"

sys_id = encode("<system>")[0]
pad_id = encode("<pad>")[0]
max_seq_len = model_config["max_seq_len"]
base_lr = 1e-4
eta_min = base_lr * 0.1

batch_size = 8
accumulate_grad_batches = 4
devices = [2,3]
training_tokens = (303 * 10**6) * 20    # model_size * 20
training_points = int(training_tokens / max_seq_len)

optim_step = accumulate_grad_batches * batch_size * len(devices)
model_save_steps = 5000
total_steps = int(training_points / optim_step)
warmup_steps = int(total_steps * 0.01)

print(f"\n{training_tokens=}, {training_points=}, {total_steps=}, {warmup_steps=}, {model_save_steps=}\n")


# Train

class DharaDataset(Dataset):
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


dataset = DharaDataset()
loader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, num_workers=15, pin_memory=True, persistent_workers=True, prefetch_factor=10)


# Val

class ValDataset(Dataset):
    def __init__(self):        
        self.all_ids = get_val_ids()
        
        
    def __len__(self):
        return len(self.all_ids)
    
    
    def __getitem__(self, index):

        id = self.all_ids[index]
        
        if id.startswith("v"):
            text = get_verse_data(id, "../ft_val_data/")
        
        if id.startswith("w"):
            text = get_word_data(id, "../ft_val_data/")
        
        if id.startswith("c"):
            text = get_chunk_data(id, "../ft_val_data/")
        
        tokens = encode(text)
        return torch.tensor(tokens, dtype=torch.long)


val_dataset = ValDataset()
val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False, num_workers=15, pin_memory=True, persistent_workers=True, prefetch_factor=10)



class LitTransf(pl.LightningModule):
    def __init__(self, model_config, base_lr, eta_min, total_steps, warmup_steps):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = TasnsfModel(**model_config)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
        
        self.val_loss_sum = 0.0
        self.val_batches = 0
        
        
    def forward(self, x):
        return self.model(x)
    
    
    def training_step(self, batch, batch_idx):
        x = batch
        
        logits = self(x)
        
        targets = x.clone()
        for i in range(x.size(0)):
            sys_pos = (x[i] == sys_id).nonzero(as_tuple=True)[0]
            if len(sys_pos) > 0:
                ans_start = sys_pos[0] + 1
                targets[i, :ans_start] = pad_id
            else:
                targets[i, :] = pad_id
        
        logits = logits[:, :-1, :]
        
        targets = x[:, 1:]
        
        targets = targets.reshape(-1)
        logits = logits.reshape(-1, logits.shape[-1])
        
        loss = self.loss_fn(logits, targets)
        
        self.log("train_loss", loss, on_step=True, prog_bar=True, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, prog_bar=True, sync_dist=True)
        
        return loss
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.base_lr)
        
        def lr_lambda(current_step):
            if current_step < self.hparams.warmup_steps:
                return current_step / max(1, self.hparams.warmup_steps)
            
            return 1.0
            

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda),
            "interval": "step"
        }
         
        return [optimizer], [scheduler]   
    
    
    def validation_step(self, batch, batch_idx):
        x = batch
        
        logits = self(x)
        
        targets = x.clone()
        for i in range(x.size(0)):
            sys_pos = (x[i] == sys_id).nonzero(as_tuple=True)[0]
            if len(sys_pos) > 0:
                ans_start = sys_pos[0] + 1
                targets[i, :ans_start] = pad_id
            else:
                targets[i, :] = pad_id
        
        targets = x[:, 1:]
        logits = logits[:, :-1, :]
        
        targets = targets.reshape(-1)
        logits = logits.reshape(-1, logits.shape[-1])
        
        loss = self.loss_fn(logits, targets)

        self.val_loss_sum += loss.detach()
        self.val_batches +=1
        
        return loss
    
    
    def on_validation_epoch_end(self):
        val_loss_sum = self.val_loss_sum
        val_batches = torch.tensor(self.val_batches, device=self.device)

        val_loss_sum = self.all_gather(val_loss_sum).sum()
        val_batches = self.all_gather(val_batches).sum()

        avg_val_loss = val_loss_sum / val_batches

        self.logger.experiment.add_scalar("val_loss_step", avg_val_loss, self.trainer.global_step)

        self.val_loss_sum = torch.tensor(0.0, device=self.device)
        self.val_batches = 0

        


checkpoint_cb = ModelCheckpoint(dirpath="pl_ft_models/checkpoints", 
                                filename="{epoch}", 
                                # monitor="val_loss", 
                                # mode="min",
                                save_top_k=-1,
                                every_n_epochs=1,
                                save_on_train_epoch_end=True
                                )

trainer = pl.Trainer(accelerator="gpu", 
                     devices=devices, 
                     strategy="ddp", 
                     precision="bf16-mixed", 
                     logger=True, 
                     log_every_n_steps=10, 
                     default_root_dir="pl_ft_models", 
                     callbacks=[checkpoint_cb], 
                     accumulate_grad_batches=accumulate_grad_batches, 
                     gradient_clip_val=1.0,
                     val_check_interval=1.0
                     )

model = LitTransf.load_from_checkpoint(ckpt_path, model_config=model_config, base_lr=5e-6, eta_min=5e-6, total_steps=None, warmup_steps=int(len(dataset) / optim_step))


trainer.fit(model, loader, val_loader)
