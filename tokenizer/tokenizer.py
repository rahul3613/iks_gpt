import json

with open("tokenizer/merges_spl.json", "r") as f:
    merges = json.load(f)


merges = [(m["x"], m["y"], m["z"]) for m in merges]

def encode(text, merges=merges):
    token_ids = list(text.lower().encode("utf-8"))
    
    for x, y, z in merges:
        
        new_token_ids = []
        i = 0
        token_ids_len = len(token_ids)
        
        while i < token_ids_len:
            if i + 1 < token_ids_len and token_ids[i] == x and token_ids[i+1] == y:
                new_token_ids.append(z)
                i += 2
            else:
                new_token_ids.append(token_ids[i])
                i += 1
        
        token_ids = new_token_ids
    return token_ids
    
    
def decode(token_ids, merges=merges):
    for x, y, z in reversed(merges):
        
        new_token_ids = []
        for token in token_ids:
            if token == z:
                new_token_ids.extend([x, y])
            else:
                new_token_ids.append(token)
        
        token_ids = new_token_ids
    return bytes(token_ids).decode()
    