import random
import json
from tqdm import tqdm


stop_words = [
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more",
    "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
    "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves", "oh", "ah", "uh", "huh", "eh",
    "um", "like", "ok", "okay", "yeah", "yes", "nah"]


def add_noise(text, noise_add_prob=1.0, char_shuff_prob=0.01, word_drop_prob=0.02, char_repl_prob=0.01, char_move_prob=0.01, space_repl_prob=0.01):
    if random.random() > noise_add_prob:
        return text
    
    text = text.lower()
    
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
        rand_rand = random.random()
        if w in stop_words:
            rand_rand /= 5
        if rand_rand < word_drop_prob:
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


data_path="../ft_data/"


def get_train_ids(data_path="../ft_data/"):
    with open(f"{data_path}verse/sampled_ids.json", "r") as f:
        verse_ids = json.load(f)
        
    with open(f"{data_path}word/sampled_ids.json", "r") as f:
        word_ids = json.load(f)
        
    with open(f"{data_path}chunk/sampled_ids.json", "r") as f:
        chunk_ids = json.load(f)
        

    print("\nVerses: ", len(verse_ids))
    print("Words: ", len(word_ids))
    print("Chunk: ", len(chunk_ids))

    all_ids = []
    for v_id in verse_ids:
        all_ids.append("v_" + str(v_id))
    
    for w_id in word_ids:
        all_ids.append("w_" + str(w_id))
    
    for c_id in chunk_ids:
        all_ids.append("c_" + str(c_id))
        
    print("\nTotal items: ", len(all_ids))
    
    return all_ids
        

def format_qna(question, answer):
    return "<user>" + add_noise(question) + "<system>" + answer + "<eos>"


def get_verse_data(id, data_path="../ft_data/"):
    id = id[2:]
    
    with open(f"{data_path}verse/{id}.json", "r") as f:
        data = json.load(f)
        
    return format_qna(data['question'], data['answer'])
        

def get_word_data(id, data_path="../ft_data/"):
    id = id[2:]
    
    with open(f"{data_path}/word/{id}.json", "r") as f:
        data = json.load(f)
        
    return format_qna(data['question'], data['answer'])


def get_chunk_data(id, data_path="../ft_data/"):
    id = id[2:]
    
    with open(f"{data_path}/chunk/{id}.json", "r") as f:
        data = json.load(f)
        
    return format_qna(data['question'], data['answer'])