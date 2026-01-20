from tokenizer.tokenizer import encode, decode
import torch
import time
from model_config import model_config
from pl_train import LitTransf

device = "cuda:3"

model = LitTransf.load_from_checkpoint(
    "pl_models/checkpoints/step=100000.ckpt",
    model_config=model_config,
    map_location=device,
)

model.eval()


def apply_repet_penalty(logits, gen_tokens, repet_penalty):
    for token in set(gen_tokens):
        logit = logits[token]
        if logit > 0:
            logits[token] = logit / repet_penalty
        else:
            logits[token] = logit * repet_penalty
            
    return logits


def top_p_filter(probs, p):
    sorted_probs, indices = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, -1)
    
    cutoff_mask = cum_probs > p
    cutoff_mask[0] = False
    sorted_probs[cutoff_mask] == 0.0
    
    filtered_probs = torch.zeros_like(probs)
    filtered_probs.scatter_(0, indices, sorted_probs)
    
    filtered_probs = filtered_probs / filtered_probs.sum()
    
    return filtered_probs


def generate(text, max_gen_len=128, temp=0.4, top_p=0.9, repet_penalty=1.2):

    inp_tokens = encode(text)
    gen_tokens = []
    next_token = -1
    eos_token = encode("<eos>")[0]

    while next_token != eos_token and len(gen_tokens) < max_gen_len:
        tokens = inp_tokens + gen_tokens
        tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        out = model(tokens)
        
        logits = out[:, -1]
        logits = logits.squeeze(0)
        logits = apply_repet_penalty(logits, gen_tokens, repet_penalty)
        logits = logits / temp
        
        probs = torch.softmax(logits, dim=-1)
        probs = top_p_filter(probs, top_p)
        next_token = torch.multinomial(probs, 1).item()
        
        
        gen_tokens.append(next_token)
        try:
            print(decode([next_token]), end="")
        except:
            pass

    return decode(gen_tokens)


text = '''english word:
dharma

word:
'''
print(text, end="")

while text != "q":
    op = generate(text)
    text = input("\n\nQuery:\n")
