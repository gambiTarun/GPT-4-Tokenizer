from .base import Tokenizer, get_stats, merge

class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
        
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        
        num_merges = vocab_size - 256
        vocab = {idx: bytes([idx]) for idx in range(256)}
        
        num_merges = vocab_size - 256
        ids = list(text.encode("utf-8"))  # copy original tokens

        merges = {}
        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get) # max on the values ie: frequency
            idx = i+256
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"{i+1}/{num_merges}, merged {pair} into new token {idx}, occurences: {stats[pair]}")
            
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()
    
    def decode(self, ids):
        text_bytes = b"".join([self.vocab[i] for i in ids])
        text = text_bytes.decode('utf-8', errors='replace')
        return text 

    def encode(self, text):
        ids = list(map(int, text.encode("utf-8")))
        for pair, idx in self.merges.items():
            ids = merge(ids, pair, idx)
        return ids
        