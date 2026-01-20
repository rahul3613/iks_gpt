import random
import json
# from tqdm import tqdm
from aksharamukha import transliterate


def aks_translit(text, to_script='Devanagari', from_script='Devanagari'):
    if from_script == to_script:
        return text
    return transliterate.process(from_script, to_script, text)


data_path="../data/"

def get_eval_ids(data_path="../data/"):
    with open(f"{data_path}verse_data/test_ids.json", "r") as f:
        verse_ids = json.load(f)
        
    with open(f"{data_path}word_data/test_ids.json", "r") as f:
        word_ids = json.load(f)
        
    with open(f"{data_path}chunk_data/test_ids.json", "r") as f:
        chunk_ids = json.load(f)
        

    # print("\nOrignal verses: ", len(verse_ids))
    
    new_verse_ids = verse_ids.copy()
    for id in verse_ids:
        with open(f"{data_path}verse_data/{id}.json", "r") as f:
            v_data = json.load(f)
            
        new_verse_ids.extend([id] * (2 * len(v_data["misc_data"])))
    # print("Final verses: ", len(new_verse_ids))
    
    
    # print("\nOrignal words: ", len(word_ids))
    
    new_word_ids = word_ids.copy()
    for id in word_ids:
        with open(f"{data_path}word_data/{id}.json", "r") as f:
            w_data = json.load(f)
            
        new_word_ids.extend([id] * (2 * max(0, (len(w_data["definitions"])-1))))
    # print("Final words: ", len(new_word_ids))
    
    # print("\nOrignal Chunk: ", len(chunk_ids))

    all_ids = []
    for v_id in new_verse_ids:
        all_ids.append("v_" + str(v_id))
    
    for w_id in new_word_ids:
        all_ids.append("w_" + str(w_id))
    
    for c_id in chunk_ids:
        all_ids.append("c_" + str(c_id))
        
    print("Total val items: ", len(all_ids))
    
    return all_ids
        

def get_verse_data(id, data_path="../data/"):
    id = id[2:]
    
    with open(f"{data_path}verse_data/{id}.json", "r") as f:
        v_data = json.load(f)
        
    parts = []
    parts.append("verse:\n" + v_data["text"])
    parts.append("reference:\n" + v_data["ref"])
    
    parts.append("IAST Verse:\n" + aks_translit(v_data["text"].replace('।', '|').replace('॥', '||'), "IAST"))
    parts.append("IAST Reference:\n" + aks_translit(v_data["ref"], "IAST"))
        
    parts.append("English Verse:\n" + aks_translit(v_data["text"].replace('।', '|').replace('॥', '||'), "RomanColloquial"))
    parts.append("English Reference:\n" + aks_translit(v_data["ref"], "RomanColloquial"))

    random.shuffle(v_data["misc_data"])
    for m_data in v_data["misc_data"][:6]:
        parts.append(m_data["title"] + ":\n" + m_data["value"])
        
    random.shuffle(parts)
    
    for m_data in v_data["misc_data"][6:]:
        parts.append(m_data["title"] + ":\n" + m_data["value"])
    
    text = "\n\n".join(parts)
    return text[:5000]


def get_word_data(id, data_path="../data/"):
    id = id[2:]
    
    with open(f"{data_path}/word_data/{id}.json", "r") as f:
        w_data = json.load(f)
        
    parts = []
    parts.append("Word:\n" + aks_translit(w_data["word"], from_script="IAST"))
    parts.append("IAST Word:\n" + w_data["word"])
    parts.append("English Word:\n" + aks_translit(w_data["word"], from_script="IAST", to_script="RomanColloquial"))
    
    random.shuffle(w_data["definitions"])
    for defn in w_data["definitions"][:3]:
        parts.append("Definition:\n" + defn)
        
    random.shuffle(parts)
        
    for defn in w_data["definitions"][3:]:
        parts.append("Definition:\n" + defn)
        
    text = "\n\n".join(parts)
    return text[:5000]


def get_chunk_data(id, data_path="../data/"):
    id = id[2:]
    
    with open(f"{data_path}/chunk_data/{id}.json", "r") as f:
        c_data = json.load(f)
        
    text = c_data["text"]
    ref = c_data["ref"]
    next = c_data["next"]
        
    parts = ["Text:\n" + text, "Reference:\n" + ref.replace("->", " -> ")]

    random.shuffle(parts)
        
    text = "\n\n".join(parts)
    return text[:5000]