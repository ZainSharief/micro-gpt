import pickle
from collections import Counter

class BPETokeniser():

    def __init__(self, num_merges: int) -> None:
        self.encode_map = {}
        self.decode_map = {}
        self.vocab_size = 0
        self.eos_token = '<|SEP|>'
        self.num_merges = num_merges 
    
    def replace(self, pairs_list: list[tuple[int]], max_pair: tuple[int], idx: int) -> list[int]:
        
        # Creating a new list is faster than inplace operations for large lists
        new_list = []
        skip = False

        for item in pairs_list: 

            # If the previous element was the max_pair, skip the current element
            if skip:
                skip = False

            elif item == max_pair:
                new_list.append(idx)
                skip = True

            else:
                new_list.append(item[0])

        # The last element should be added if the last pair is the max_pair otherwise it is missed
        if not skip: 
            new_list.append(pairs_list[-1][1])

        return new_list

    def create_encoding(self, data: str, min_freq: int = 5, file_path: str = None) -> list[int]:

        # Converts the data to utf-8 values
        data = list(map(int, data.encode('utf-8')))
        idx = max(data) + 1
    
        for num in range(self.num_merges):

            # Has to recalculate the pairs each time to account for new pairs
            pairs_list = list(zip(data[:-1], data[1:]))
            pair_counts = Counter(pairs_list)

            # Validates a pair appears more than min_freq times
            if max(pair_counts.values()) < min_freq:
                print(f'No pair is present more than {min_freq} times. Terminating...')
                break
            max_pair = max(pair_counts, key=pair_counts.get)
            
            data = self.replace(pairs_list, max_pair, idx)
            self.encode_map[max_pair] = idx
            idx += 1

            print(f'\rTokeniser:{num+1}/{self.num_merges} complete', end='')

        # Decode map is a swapped version of the encode map
        self.decode_map = {k:v for v,k in self.encode_map.items()}

        # Map only holds the pairs -> not utf-8 values
        self.vocab_size = max(data) + 1 

        if file_path:
            self.save(file_path)

        return data

    def encode(self, data: str) -> list[int]:

        data = list(map(int, data.encode('utf-8')))

        for encoding_pair in self.encode_map.keys():
            pairs_list = list(zip(data[:-1], data[1:]))
            idx = self.encode_map[encoding_pair]
            data = self.replace(pairs_list, encoding_pair, idx)
        
        return data 
    
    def decode(self, tokens: list[int]) -> str:

        # Should replace the tokens in reverse order as this accounts for "nested pairs"
        # Dictionaries are ordered in python 3.7+ 
        for decode_idx in sorted(self.decode_map.keys(), reverse=True):
            max_pair = self.decode_map[decode_idx]

            for i, item in enumerate(tokens):
                tokens[i:i+1] = max_pair if item == decode_idx else tokens[i:i+1]

        tokens = bytes(tokens)
        tokens = tokens.decode('utf-8', errors="replace")

        return tokens   

    def save(self, file_path: str) -> None:
        with open(file_path, "wb") as f:
            # Only need to save the encode map as decode map can be recreated
            pickle.dump((self.encode_map, self.vocab_size), f)

    def load(self, file_path: str) -> bool:
        try:
            with open(file_path, "rb") as f:
                self.encode_map, self.vocab_size = pickle.load(f)

                # Recreating the decode map
                self.decode_map = {k:v for v,k in self.encode_map.items()}

            print('Successfully loaded the tokeniser.')
            return True
        
        except:
            print('Could not load the tokeniser.')
            return False

if __name__ == '__main__':
    tokeniser = BPETokeniser(20_000)
    tokeniser.load('byte-pair-encoding20000.pkl')
    test_text = "Hi, how are you?"
    tokens = tokeniser.encode(test_text)
    print(f"Tokens: {tokens}")
    print(f"Decoded: {tokeniser.decode(tokens)}")