from transformers import GPT2Tokenizer

class GPTtokenizer():

    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=True)
        self.vocab_size = self.tokenizer.vocab_size

    def encode(self, data):
        return self.tokenizer.encode(data, return_tensors="pt")

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)