from collections import Counter
import json
import regex as re
import time
from tqdm import tqdm
from aksharamukha import transliterate


def aks_translit(text, to_script='Devanagari', from_script='Devanagari'):
    if from_script == to_script:
        return text
    return transliterate.process(from_script, to_script, text)


PATTERN = re.compile(
    r"\n+"
    r"| 's|'t|'re|'ve|'m|'ll|'d"
    r"| ?(?:\p{L}\p{M}*)+"
    r"| ?\p{N}+"
    r"| ?[^\s\p{L}\p{M}\p{N}]+"
    r"|[ \t]+(?!\S)"
    r"|[ \t]+"
)


data_path = "../../data"

with open(f"{data_path}/verse_data/verse_ids.json", "r") as f:
    verse_ids = json.load(f)
    
with open(f"{data_path}/word_data/word_ids.json", "r") as f:
    word_ids = json.load(f)
    
with open(f"{data_path}/chunk_data/chunk_ids.json", "r") as f:
    chunk_ids = json.load(f)
    
    
vocab_dict = {}
    
    
print("Loading Verse Data:")
for id in tqdm(verse_ids):
    with open(f"{data_path}/verse_data/{id}.json", "r") as f:
        v_data = json.load(f)
    
    text = "Verse:\n" + v_data["text"] + "\n\nReference:\n" + v_data["ref"]
    text += "\n\nIAST Verse:\n" + aks_translit(v_data["text"].replace('।', '|').replace('॥', '||'), "IAST") + "\n\nIAST Reference:\n" + aks_translit(v_data["ref"], "IAST")
    for m_data in v_data["misc_data"]:
        text += "\n\n" + m_data["title"] + ":\n" + m_data["value"]

    text = text.lower()
    parts = PATTERN.findall(text)

    for part in parts:
        try:
            vocab_dict[part]["freq"] += 1
        except KeyError:
            vocab_dict[part] = {"freq":1, "token_ids": list(part.encode('utf-8'))}
        
        
print("Loading Word Data:")
for id in tqdm(word_ids):
    with open(f"{data_path}/word_data/{id}.json", "r") as f:
        w_data = json.load(f)
    
    text = "Word:\n" + aks_translit(w_data["word"], from_script="IAST") + "\n\nIAST Word:\n" + w_data["word"]
    for defn in w_data["definitions"]:
        text += "\n\nDefinition:\n" + defn

    text = text.lower()
    parts = PATTERN.findall(text)

    for part in parts:
        try:
            vocab_dict[part]["freq"] += 1
        except KeyError:
            vocab_dict[part] = {"freq":1, "token_ids": list(part.encode('utf-8'))}        
            
            
print("Loading Chunk Data:")
for id in tqdm(chunk_ids):
    with open(f"{data_path}/chunk_data/{id}.json", "r") as f:
        c_data = json.load(f)
    
    text = "Text:\n" + c_data["text"] + "\n\nReference:\n" + c_data["ref"]

    text = text.lower()
    parts = PATTERN.findall(text)

    for part in parts:
        try:
            vocab_dict[part]["freq"] += 1
        except KeyError:
            vocab_dict[part] = {"freq":1, "token_ids": list(part.encode('utf-8'))}
            
        
print("Total Words", len(vocab_dict))
# 94,20,601
# print(vocab_dict)



def update_list(token_ids, x, y, curr_vocab_length):
    new_list = []
    i = 0
    
    while i < len(token_ids):
        if token_ids[i:i+2] == [x,y]:
            new_list.append(curr_vocab_length)
            i += 2
        else:
            new_list.append(token_ids[i])
            i += 1
            
    return new_list



vocab_length = 25000
curr_vocab_length = 255
merges = []
o_time = time.time()
avg_time = 0

freq_counter = Counter()
pair_words_dict = {}

for word, value in tqdm(vocab_dict.items()):
    token_ids = value["token_ids"]
    list_len = len(token_ids)
    
    if list_len > 1:
        for i in range(list_len-1):
            token_key = (token_ids[i], token_ids[i+1])
            freq_counter[token_key] += value["freq"]
            
            v = pair_words_dict.get(token_key)
            if v is None:
                pair_words_dict[token_key] = set([word])
            else:
                v.add(word)
                

while curr_vocab_length < vocab_length:
                                
    (x,y), freq = freq_counter.most_common(1)[0]

    curr_vocab_length += 1

    n_time = time.time()
    avg_time = 0.9 * avg_time + 0.1 * (n_time - o_time)
    print("x:", x, "  y:", y, "  z:", curr_vocab_length, "  freq:", freq, "  time:", round(avg_time, 2), "  eta:", round(avg_time * (vocab_length - curr_vocab_length), 0))
    o_time = n_time
    

    words = pair_words_dict[(x, y)]
    for word in words:
        voc_val = vocab_dict[word]
        voc_tok_ids = voc_val["token_ids"]
        
        new_token_ids = update_list(voc_tok_ids, x, y, curr_vocab_length)
        voc_val["token_ids"] = new_token_ids
        
        if len(new_token_ids) > 1:
            for i in range(0, len(new_token_ids)):
                if new_token_ids[i] == curr_vocab_length:
                    if i > 0:
                        w = new_token_ids[i-1]
                        freq_counter[(w, x)] -= voc_val["freq"]
                        
                        token_key = (w, curr_vocab_length)
                        freq_counter[token_key] += voc_val["freq"]
                        
                        v = pair_words_dict.get(token_key)
                        if v is None:
                            pair_words_dict[token_key] = set([word])
                        else:
                            v.add(word)
                        
                        
                    if i < len(new_token_ids) - 1:
                        z = new_token_ids[i+1]
                        freq_counter[(y, z)] -= voc_val["freq"]
                                           
                        if curr_vocab_length != z:     
                            token_key = (curr_vocab_length, z)
                            freq_counter[token_key] += voc_val["freq"]
                            
                            v = pair_words_dict.get(token_key)
                            if v is None:
                                pair_words_dict[token_key] = set([word])
                            else:
                                v.add(word)
        

    del freq_counter[(x, y)]
    del pair_words_dict[(x, y)]
    

    merges.append({"x": x, "y": y, "z": curr_vocab_length})

    if curr_vocab_length % 1000 == 0:
        with open(f"merges_{curr_vocab_length}.json", "w") as f:
            json.dump(merges, f)

with open("merges.json", "w") as f:
    json.dump(merges, f)
