import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tokenizer.tokenizer import encode

eos_id = encode("<eos>")[0]

class SelfAttn(nn.Module):
    def __init__(self, embd_dim, head_dim):
        super(SelfAttn, self).__init__()
        
        self.head_dim = head_dim
        
        self.Q = nn.Linear(embd_dim, head_dim, bias=False)
        self.K = nn.Linear(embd_dim, head_dim, bias=False)
        self.V = nn.Linear(embd_dim, head_dim, bias=False)
        
        self.attn_dropout = nn.Dropout(0.1)
        
        
    def forward(self, x, mask):
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        
        score = Q @ K.transpose(-2, -1)
        score = score.masked_fill(mask==0, -1e9)
        score /= self.head_dim ** 0.5
        
        attn = torch.softmax(score, dim=-1)
        attn = self.attn_dropout(attn)
        attn_out = attn @ V
        
        return attn_out
    
    
    
class MultiHeadAttn(nn.Module):
    def __init__(self, embd_dim, num_head):
        super(MultiHeadAttn, self).__init__()
        
        assert embd_dim % num_head == 0, "embd_dim must be divisible by num_heads"
        head_dim = embd_dim // num_head
        
        self.heads = nn.ModuleList([SelfAttn(embd_dim, head_dim) for _ in range(num_head)])
        self.out_proj = nn.Linear(embd_dim, embd_dim, bias=False)
        
        self.dropout = nn.Dropout(0.1)
        
        
    def forward(self, x, mask):
        head_outputs = [head(x, mask) for head in self.heads]
        out = torch.cat(head_outputs, dim=-1)
        out = self.out_proj(out)
        
        return self.dropout(out)



class TransfBlock(nn.Module):
    def __init__(self, embd_dim, num_head):
        super(TransfBlock, self).__init__()

        self.multi_head_attn = MultiHeadAttn(embd_dim, num_head)
        
        self.dropout = nn.Dropout(p=0.2)
        
        self.ln1 = nn.LayerNorm(embd_dim)
        self.ln2 = nn.LayerNorm(embd_dim)
        
        self.up = nn.Linear(embd_dim, 4 * embd_dim, bias=False)
        self.gate = nn.Linear(embd_dim, 4 * embd_dim, bias=False)
        self.down = nn.Linear(4 * embd_dim, embd_dim, bias=False)
        

    def forward(self, x, mask):

        # Attention
        x_norm = self.ln1(x)        
        attn_out = self.multi_head_attn(x_norm, mask)
        x = attn_out + x                        # Resedual connection
        
        # FeedForward
        x_norm2 = self.ln2(x)
        out = F.silu(self.gate(x_norm2)) * self.up(x_norm2)     # (, seq_len, embd_dim)  ->  (, seq_len, 4*embd_dim)
        out = self.dropout(out)
        out = self.down(out)                     # (, seq_len, 4*embd_dim)  ->  (, seq_len, embd_dim)
        
        out = out + x
        
        return out
    
    
def get_mask(x):
    seq_len = x.shape[-1]
    
    mask = torch.zeros((x.shape[0], seq_len, seq_len), device=x.device)

    for i in range(x.shape[0]):
        eos_indexes = (torch.where(x[i] == eos_id)[0] + 1).tolist()
        prev_index = 0
        if (not eos_indexes) or (eos_indexes[-1] != seq_len):
            eos_indexes.append(seq_len)

        for index in eos_indexes:
            index_len = index - prev_index
            curr_mask = torch.tril(torch.ones(index_len, index_len))
            mask[i, prev_index:prev_index + index_len, prev_index:prev_index + index_len] = curr_mask
            
            prev_index = index
                    
    return mask
    
    
class TasnsfModel(nn.Module):
    def __init__(self, vocab_size, max_seq_len, embd_dim, num_head, num_layers, doc_masking=False, tie_embeddings=False):
        super(TasnsfModel, self).__init__()
        
        self.doc_masking = doc_masking
        self.embd_dim = embd_dim
        
        self.embedding = nn.Embedding(vocab_size, embd_dim)
        self.pos_embd = nn.Embedding(max_seq_len, embd_dim)
        
        self.transf_blocks = nn.ModuleList([TransfBlock(embd_dim, num_head) for _ in range(num_layers)])
        
        self.ln_f = nn.LayerNorm(embd_dim)
        
        self.op_proj = nn.Linear(embd_dim, vocab_size, bias=False)
        if tie_embeddings:
            self.op_proj.weight = self.embedding.weight
        
        
    def forward(self, x):
        
        seq_len = x.shape[-1]
        positions = torch.arange(seq_len, device=x.device)  # -> (seq_len)
        
        if self.doc_masking:
            mask = get_mask(x)
        else:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0) # Lower triangular matrix with -> (seq_len, seq_len) -> (1, seq_len, seq_len)
        
        pos_embd = self.pos_embd(positions).unsqueeze(0)        # (seq_len)  ->  (seq_len, embd_dim)  ->  (1, seq_len, embd_dim)
        x = self.embedding(x) + pos_embd        # (batch, seq_len)  ->  (batch, seq_len, embd_dim)
        
        for block in self.transf_blocks:
            x = block(x, mask)
        
        out = self.ln_f(x)
        logits = self.op_proj(out)              # (, seq_len, embd_dim)  ->  (, seq_len, op_vocab)
        
        return logits
    