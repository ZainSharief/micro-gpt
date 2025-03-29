from transformers import GPT2Tokenizer

class GPTtokenizer():

    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=True, verbose=False)
        self.vocab_size = self.tokenizer.vocab_size
        self.eos_token = self.tokenizer.eos_token_id
        self.pad_token_id = 0

    def encode(self, data):
        return self.tokenizer.encode(data, return_tensors="pt")

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, errors="replace")