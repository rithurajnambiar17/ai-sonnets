import pickle

def depickle_tokenizer(path):
    with open(path, 'rb') as handle:
        tokenizerLoaded=pickle.load(handle)
    return tokenizerLoaded