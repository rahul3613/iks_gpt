import random
import json
from tqdm import tqdm
from aksharamukha import transliterate


def aks_translit(text, to_script='Devanagari', from_script='Devanagari'):
    if from_script == to_script:
        return text
    return transliterate.process(from_script, to_script, text)


def shuff_drop(text, drop_prop=0.05, shuffle_prob = 0.05):
    if random.random() < drop_prop:
        return ""
    
    if random.random() < shuffle_prob:
        add_nl = False
        if text.endswith(":\n"):
            text = text[:-2]
            add_nl = True
            
        words = text.split(" ")
        random.shuffle(words)
        text = " ".join(words)
        
        if add_nl:
            text = text + ":\n"
    
    return add_noise(text)


def add_noise(text, noise_add_prob=0.1, char_shuff_prob=0.005, word_drop_prob=0.01, char_repl_prob=0.005, char_move_prob=0.005, space_repl_prob=0.005):
    if random.random() > noise_add_prob:
        return text
    
    # char shuffle
    if len(text) > 1:
        chars = list(text)
        for i in range(len(chars)-1):
            if random.random() < char_shuff_prob:
                chars[i], chars[i+1] = chars[i+1], chars[i]
            
        text = "".join(chars)
    
    # word drop
    words = text.split(" ")
    new_words = []
    for w in words:
        if random.random() < word_drop_prob:
            continue
        new_words.append(w)

    text = " ".join(new_words)
    
    # char repl
    text = list(text)
    len_text = len(text)

    for i in range(len_text):
        if random.random() < char_repl_prob:
            text[random.sample(range(len_text), 1)[0]] = random.sample(text, 1)[0]
        
    text = "".join(text)
    
    # char displ
    text = list(text)
    n = len(text)
    
    k = int(n * char_move_prob)
    if k >= 1:
        remove_idxs = sorted(random.sample(range(n), k))
        removed_chars = [text[i] for i in remove_idxs]

        for i in reversed(remove_idxs):
            text.pop(i)
            
        for ch in removed_chars:
            insert_pos = random.randint(0, len(text))
            text.insert(insert_pos, ch)

    text = "".join(text)
    
    # space replace
    out = []
    for c in text:
        if (c == " " or c == "\n") and random.random() < space_repl_prob:
            pass
        else:
            out.append(c)
        
        if random.random() < space_repl_prob / 5:
            out.append(" ")
        if random.random() < space_repl_prob / 10:
            out.append("\n")
    
    return "".join(out)


data_path="../../data/"

with open(f"{data_path}chunk_data/test_ids.json", "r") as f:
    test_chunk_ids = json.load(f)

def get_train_ids(data_path=data_path):
    with open(f"{data_path}verse_data/train_ids.json", "r") as f:
        verse_ids = json.load(f)
        
    with open(f"{data_path}word_data/train_ids.json", "r") as f:
        word_ids = json.load(f)
        
    with open(f"{data_path}chunk_data/train_ids.json", "r") as f:
        chunk_ids = json.load(f)
        

    print("\nOrignal verses: ", len(verse_ids))
    
    new_verse_ids = verse_ids.copy()
    for id in tqdm(verse_ids, desc="Updating verse ids"):
        with open(f"{data_path}verse_data/{id}.json", "r") as f:
            v_data = json.load(f)
            
        new_verse_ids.extend([id] * (2 * len(v_data["misc_data"])))
    print("Final verses: ", len(new_verse_ids))
    
    
    print("\nOrignal words: ", len(word_ids))
    
    new_word_ids = word_ids.copy()
    for id in tqdm(word_ids, desc="Updating word ids"):
        with open(f"{data_path}word_data/{id}.json", "r") as f:
            w_data = json.load(f)
            
        new_word_ids.extend([id] * (2 * max(0, (len(w_data["definitions"])-1))))
    print("Final words: ", len(new_word_ids))
    
    print("\nOrignal Chunk: ", len(chunk_ids))

    all_ids = []
    for v_id in new_verse_ids:
        all_ids.append("v_" + str(v_id))
    
    for w_id in new_word_ids:
        all_ids.append("w_" + str(w_id))
    
    for c_id in chunk_ids:
        all_ids.append("c_" + str(c_id))
        
    print("Total items: ", len(all_ids))
    
    return all_ids
        

def get_verse_data(id, data_path=data_path):
    id = id[2:]
    
    with open(f"{data_path}verse_data/{id}.json", "r") as f:
        v_data = json.load(f)
        
    parts = []
    parts.append(shuff_drop("verse:\n") + v_data["text"])
    parts.append(shuff_drop("reference:\n") + v_data["ref"])
    
    parts.append(shuff_drop("IAST Verse:\n") + aks_translit(v_data["text"].replace('।', '|').replace('॥', '||'), "IAST"))
    parts.append(shuff_drop("IAST Reference:\n") + aks_translit(v_data["ref"], "IAST"))
        
    parts.append(shuff_drop("English Verse:\n") + aks_translit(v_data["text"].replace('।', '|').replace('॥', '||'), "RomanColloquial"))
    parts.append(shuff_drop("English Reference:\n") + aks_translit(v_data["ref"], "RomanColloquial"))

    random.shuffle(v_data["misc_data"])
    for m_data in v_data["misc_data"][:6]:
        parts.append(shuff_drop(m_data["title"] + ":\n") + m_data["value"])
        
    random.shuffle(parts)
    
    for m_data in v_data["misc_data"][6:]:
        parts.append(shuff_drop(m_data["title"] + ":\n") + m_data["value"])
    
    text = "\n\n".join(parts)
    return text[:5000]


def get_word_data(id, data_path=data_path):
    id = id[2:]
    
    with open(f"{data_path}/word_data/{id}.json", "r") as f:
        w_data = json.load(f)
        
    parts = []
    parts.append(shuff_drop("Word:\n") + aks_translit(w_data["word"], from_script="IAST"))
    parts.append(shuff_drop("IAST Word:\n") + w_data["word"])
    parts.append(shuff_drop("English Word:\n") + aks_translit(w_data["word"], from_script="IAST", to_script="RomanColloquial"))
    
    random.shuffle(w_data["definitions"])
    for defn in w_data["definitions"][:3]:
        parts.append(shuff_drop("Definition:\n") + defn)
        
    random.shuffle(parts)
        
    for defn in w_data["definitions"][3:]:
        parts.append(shuff_drop("Definition:\n") + defn)
        
    text = "\n\n".join(parts)
    return text[:5000]


def get_chunk_data(id, data_path=data_path):
    id = id[2:]
    
    with open(f"{data_path}/chunk_data/{id}.json", "r") as f:
        c_data = json.load(f)
        
    text = c_data["text"]
    ref = c_data["ref"]
    next = c_data["next"]
    
    parts = f"{shuff_drop("Reference:\n") + ref.replace("->", " -> ")}\n\n{shuff_drop("Text:\n") + text}"
    count = 0
    while next and count < 6:
        count += 1
        try:
            with open(f"{data_path}/chunk_data/{next}.json", "r") as f:
                c_data = json.load(f)
                
            if next not in test_chunk_ids:
                if ref == c_data["ref"]:
                    parts += "\n" + c_data["text"]
                else:
                    parts += f"\n\n{shuff_drop("Reference:\n") + c_data["ref"].replace("->", " -> ")}\n\n{shuff_drop("Text:\n") + c_data["text"]}"
                    ref = c_data["ref"]

            next = c_data["next"]
        except:
            continue
        
    return parts[:5000]