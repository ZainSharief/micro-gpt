import pickle
from collections import Counter

class BPETokeniser():

    def __init__(self, num_merges: int) -> None:
        self.encode_map = {}
        self.decode_map = {}
        self.vocab_size = 0
        self.eos_token = ord('\n')
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
    
        for _ in range(self.num_merges):

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
    import timeit

    # Wrapper function for timing
    def time_create_encoding(tokeniser_class, data):
        tokeniser = tokeniser_class(1000)
        tokeniser.create_encoding(data)

    # The data for testing
    data = """2MGKwAS1PzI4z6hs1kuDX9OYIwKhkJPG7a5LG3CqZnQjbpCanhqeAClXv5HtuNfIRQ48qZNQWAL41R7PR8h5oh5QAEeIRRdIb7rHmAVO0DlEQjBhgTcRCsCzQ4F18PcMypsCntbzjN0UuUklUyONM7NNeqiaqsYEq0DDvwrZCBvlfTo5QqVm52yuihh1Ti4dSRAnrrDo
    Rh3A4Teu3pz6NYwJFd2FbX9fDjD12Yi9CDM5MIHK5R01pwlQCiW4iZWfA4BZc1OLyQpnUZYrkARN6AVT6n6Zp4IAjbMn2o1OGUP6Zg0L1vJuRUdIIAdqeji322YQjyTDyKbcAkzxpncFhYPWvRuw44TKQPGV3AGoXkoAz 94x04sq1zq6GvST7Bva38NViHmeJZocbtT
    Vdal5TNnF7F s5tB4AOzosAN7lsWPlMolMTLUG53EluCwyFMvSXUtcaXq0GuuZnzsS6NZLtaxIp1e4RzeIapAd7lJImSWNYVceB6bvqtjBMRXeypbzNImSEBAo8nD9W fjXs6Q7roclOdNc6PS1WiHkpsirF1e48mLyYM3hgmMEsIKU3OQJjA0JSsG pWagL9Xjl PVP
    6OWctzzcqrSY1d2dSxejZ0G3Vwuiao0KkQL1d6BYeTBrPxqsEavDLKuMQBLD4TpzXFBQRQ81YLFwklR91juhZ1YoVo7FWnNvtQJvTT3JH31 S1YIVBwqaSrC6C5jXQhh6 k9nnlYgw7kaD2R9wjAA4zJIAnNnDVg2vVoMbpZ3eJSDEoWtq6aSABGu96NKs3QuYX6N88F
    DlsxbTXv020znL1qTGfaVuDNQ3yZHLWm2ljJu4lg280PLQ2BKuOINFw 1Lb7a1a7Z1g1sDxswlli5OIrlzWlFAl2KJNhxkjJREcUINsYghbX9pCbFXbgHihZTcITe9F918qw8xTXKXJvj4h3neBSePP7eekY0o N1SZPIHN2yJsAGgAAorxZGdGB2oXXYTUsbic47ad8"""

    # Timing BPETokeniser
    time_tokeniser = timeit.timeit(
        stmt="time_create_encoding(BPETokeniser, data)",
        globals=globals(),
        number=10 
    )

    print(f"BPETokeniser time: {time_tokeniser:.6f} seconds (average over 10 runs)")