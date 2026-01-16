from basic import TasnsfModel
from model_config import model_config

import torch
import torch.nn as nn
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


max_seq_len = model_config["max_seq_len"]
base_lr = 1e-4
eta_min = base_lr * 0.1

batch_size = 8
accumulate_grad_batches = 4
devices = [2,3]

optim_step = accumulate_grad_batches * batch_size * len(devices)
model_save_steps = int(100000 / optim_step)
total_steps = int(3000000 / optim_step)
warmup_steps = int(total_steps * 0.01)

print(f"\n{total_steps=}, {warmup_steps=}, {model_save_steps=}\n")


class DharaDataset(Dataset):
    def __init__(self, verse_prob=0.4, word_prob=0.3, chunk_prob=0.3):
        
        self.verse_prob = verse_prob
        self.word_prob = word_prob
        self.chunk_prob = chunk_prob
        
        self.verse_tokens = np.memmap("../data/verse_train_tokens.bin", dtype=np.uint16, mode="r")
        self.word_tokens = np.memmap("../data/word_train_tokens.bin", dtype=np.uint16, mode="r")
        self.chunk_tokens = np.memmap("../data/chunk_train_tokens.bin", dtype=np.uint16, mode="r")
        
        
    def __len__(self):
        return 5 * 10**6
    
    
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



class LitTransf(pl.LightningModule):
    def __init__(self, model_config, base_lr, eta_min, total_steps, warmup_steps):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = TasnsfModel(**model_config)
        
        self.loss_fn = nn.CrossEntropyLoss()
        
        
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


checkpoint_cb = ModelCheckpoint(dirpath="pl_models/checkpoints", filename="{step}", save_top_k=-1, every_n_train_steps=model_save_steps)

trainer = pl.Trainer(accelerator="gpu", devices=devices, strategy="ddp", precision="bf16-mixed", logger=True, log_every_n_steps=10, default_root_dir="pl_models", callbacks=[checkpoint_cb], accumulate_grad_batches=accumulate_grad_batches, gradient_clip_val=1.0)

model = LitTransf(model_config, base_lr=base_lr, eta_min=eta_min, total_steps=total_steps, warmup_steps=warmup_steps)

trainer.fit(model, loader)
