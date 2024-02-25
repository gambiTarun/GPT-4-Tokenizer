import unicodedata

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i<len(ids):
        if i<len(ids)-1 and ids[i]==pair[0] and ids[i+1]==pair[1]:
            new_ids.append(idx)
            i+=2
        else:
            new_ids.append(ids[i])
            i+=1
    return new_ids

def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

############################################################

class Tokenizer():
    def __init__(self):
        self.special_tokens = {}
        self.merges = {}
        self.vocab = self._build_vocab()
        self.pattern = "" # regex pattern for categorization before tokenization
        
    def train(self, text, vocab_size):
        raise NotImplementedError

    def encode(self, text):
        raise NotImplementedError

    def decode(self, ids):
        raise NotImplementedError

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special_token, idx in self.special_tokens.items():
            vocab[idx] = special_token.encode('utf-8')
        return vocab
    
    def save(self, prefix):
        """ 
        Save two files: prefix.vocab and prefix.model 
        - model file is the main file that contains the merges
        - vocab is just for inspection and human readability
        """
        with open(prefix+'.vocab', 'w') as f:
            f.write('bpe v1\n')
            f.write(f'{self.pattern}\n')
            f.write(f'{len(self.special_tokens)}\n')
            for token, idx in self.special_tokens.items():
                f.write(f'{token} {idx}\n')
            for idx1, idx2 in self.merges.items():
                f.write(f'{idx1} {idx2}\n')
            
        inverse_vocab = {v: k for k, v in self.merges.items()}
        with open(prefix+'.model', 'w') as f:
            for idx, token in self.vocab.items():
                s = render_token(token)
                if idx in inverse_vocab:
                    idx0, idx1 = inverse_vocab[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    
                    f.write(f'[{s0}][{s1}] -> [{s}] {idx}\n')
                else:
                    f.write(f'[{s}] {idx}\n')            
    
    def load(self, file):
        """
        Load just using the .model file
        """
        assert file.endswith('.model')
        
        special = {}
        merges = {}
        idx = 256
        with open(file, 'r', encoding='utf-8') as f:
            
            version = f.readline().strip()
            assert version == 'bpe v1'
                
            pattern = f.readline().strip()
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                token, idx = f.readline().strip().split()
                special[token] = int(idx)
            
            for line in f:
                idx1, idx2 = map(int, line.strip().split())
                merges[(idx1, idx2)] = idx
                idx += 1
            
        self.special_tokens = special
        self.merges = merges
        self.pattern = pattern
        