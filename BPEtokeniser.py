import pickle
from collections import Counter

class BPETokeniser():

    def __init__(self, num_merges):
        self.encode_map = {}
        self.decode_map = {}
        self.vocab_size = 0
        self.num_merges = num_merges

    def replace(self, data, max_pair, idx):
        new_list = []
        x = 0
        while x < len(data):
            if  data[x] == max_pair[0] and x < len(data) - 1 and data[x+1] == max_pair[1]:
                new_list.append(idx)
                x += 2
            else:
                new_list.append(data[x])
                x += 1

        return new_list

    def create_encoding(self, data, min_freq=5, file_path=None):
        data = list(map(int, data.encode('utf-8')))
        idx = max(data) + 1
    
        for _ in range(self.num_merges):
            pairs_list = list(zip(data[:-1], data[1:]))
            pair_counts = Counter(pairs_list)
            if max(pair_counts.values()) < min_freq:
                print(f'No pair is present more than {min_freq} times. Terminating...')
                break
            max_pair = max(pair_counts, key=pair_counts.get)
            
            data = self.replace(data, max_pair, idx)
            self.encode_map[max_pair] = idx
            self.decode_map[idx] = max_pair
            idx += 1

        self.vocab_size = max(data) + 1
        if file_path:
            self.save(file_path)
        return data

    def encode(self, data):
        data = list(map(int, data.encode('utf-8')))

        for encoding_pair in self.encode_map.keys():
            idx = self.encode_map[encoding_pair]
            data = self.replace(data, encoding_pair, idx)
        
        return data 
    
    def decode(self, tokens):

        for decode_idx in sorted(self.decode_map.keys(), reverse=True):
            max_pair = self.decode_map[decode_idx]

            for i, item in enumerate(tokens):
                tokens[i:i+1] = max_pair if item == decode_idx else tokens[i:i+1]

        tokens = bytes(tokens)
        tokens = tokens.decode('utf-8', errors="replace")

        return tokens   

    def save(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump((self.encode_map, self.decode_map, self.vocab_size), f)

    def load(self, file_path):
        try:
            with open(file_path, "rb") as f:
                self.encode_map, self.decode_map, self.vocab_size = pickle.load(f)
            print('Successfully loaded the tokeniser.')
            return True
        except:
            print('Could not load the tokeniser.')
            return False
        
if __name__ == '__main__':
    tokeniser_path = 'BPEtokeniser.pkl'

    tokeniser = BPETokeniser(num_merges=10_000)
    tokeniser.load(tokeniser_path)

    print(tokeniser.encode('You mean I am groaned a few words?'))
    print(tokeniser.encode("You are right. Our company’s Christmas party has always been an amazing occasion for everybody."))

