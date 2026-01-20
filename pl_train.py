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
from val_utils import get_eval_ids, get_verse_data, get_chunk_data, get_word_data


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
model_save_steps = int(320000 / optim_step)
total_steps = int(training_points / optim_step)
warmup_steps = int(total_steps * 0.01)

print(f"\n{training_tokens=}, {training_points=}, {total_steps=}, {warmup_steps=}, {model_save_steps=}\n")


# Train

class DharaDataset(Dataset):
    def __init__(self, training_points, verse_prob=0.4, word_prob=0.3, chunk_prob=0.3):
        
        self.training_points = training_points
        
        self.verse_prob = verse_prob
        self.word_prob = word_prob
        self.chunk_prob = chunk_prob
        
        self.verse_tokens = np.memmap("../data/verse_train_tokens.bin", dtype=np.uint16, mode="r")
        self.word_tokens = np.memmap("../data/word_train_tokens.bin", dtype=np.uint16, mode="r")
        self.chunk_tokens = np.memmap("../data/chunk_train_tokens.bin", dtype=np.uint16, mode="r")
        
        
    def __len__(self):
        return self.training_points
    
    
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


dataset = DharaDataset(training_points, verse_prob=0.5, word_prob=0.35, chunk_prob=0.15)
loader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2)


# Val

class ValDataset(Dataset):
    def __init__(self):        
        self.all_ids = get_eval_ids()
        
        
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


def val_collate_fn(batch):
    lengths = [len(b) for b in batch]
    max_len = min(max(lengths), max_seq_len)
    
    padded = []
    for b in batch:
        if len(b) < max_len:
            b = torch.cat([b, torch.full((max_len - len(b),), pad_id)])
        else:
            if random.random() < 0.9:
                b = b[:max_len]
            else:
                b = b[-max_len:]
        padded.append(b)
            
    return torch.stack(padded)


val_dataset = ValDataset()
val_loader = DataLoader(val_dataset, collate_fn=val_collate_fn, batch_size=batch_size, shuffle=False, num_workers=15, pin_memory=True, persistent_workers=True, prefetch_factor=10)



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
            
            if current_step > self.hparams.total_steps:
                return self.hparams.eta_min / self.hparams.base_lr

            progress = (current_step - self.hparams.warmup_steps) / max(1, (self.hparams.total_steps - self.hparams.warmup_steps))
            return max((self.hparams.eta_min / self.hparams.base_lr), 0.5 * (1.0 + math.cos(progress * math.pi)))

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda),
            "interval": "step"
        }
         
        return [optimizer], [scheduler]   
    
    
    def validation_step(self, batch, batch_idx):
        x = batch
        
        logits = self(x)
        
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

        self.log("val_loss", avg_val_loss, prog_bar=True, sync_dist=False)
        self.logger.experiment.add_scalar("val_loss_step", avg_val_loss, self.trainer.global_step)

        self.val_loss_sum = torch.tensor(0.0, device=self.device)
        self.val_batches = 0

        


checkpoint_cb = ModelCheckpoint(dirpath="pl_models/checkpoints", 
                                filename="{step}_{val_loss:.3f}", 
                                monitor="val_loss", 
                                mode="min",
                                save_top_k=10, 
                                every_n_train_steps=model_save_steps
                                )

trainer = pl.Trainer(accelerator="gpu", 
                     devices=devices, 
                     strategy="ddp", 
                     precision="bf16-mixed", 
                     logger=True, 
                     log_every_n_steps=10, 
                     default_root_dir="pl_models", 
                     callbacks=[checkpoint_cb], 
                     accumulate_grad_batches=accumulate_grad_batches, 
                     gradient_clip_val=1.0,
                     val_check_interval=model_save_steps
                     )

model = LitTransf(model_config, base_lr=base_lr, eta_min=eta_min, total_steps=total_steps, warmup_steps=warmup_steps)

trainer.fit(model, loader, val_loader)
