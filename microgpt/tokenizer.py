from transformers import GPT2Tokenizer

class GPTtokenizer():

    def __init__(self, max_length: int = 385):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=True, verbose=False, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = "left"
    
        new_tokens = {
            'user_token': '<|user|>',
            'end_user_token': '<|enduser|>',
            'assistant_token': '<|assistant|>',
            'end_assistant_token': '<|endassistant|>'
        }
        self.add_tokens(new_tokens)
        
        self.max_length = max_length
        self.vocab_size = self.tokenizer.vocab_size

        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        
    def add_tokens(self, tokens):
        self.tokenizer.add_tokens(list(tokens.values()))
        for name, token in tokens.items():
            setattr(self, name, token)

        for name, token in tokens.items():
            setattr(self, f'{name}_id', int(self.tokenizer.convert_tokens_to_ids(token)))
    
    def encode(self, data):
        return self.tokenizer(data, return_tensors="pt", add_special_tokens=False)["input_ids"]
    
    def encode_padding(self, data):
        return self.tokenizer(data, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")["input_ids"].squeeze(0)
 
    def decode(self, tokens):
        return self.tokenizer.decode(tokens, errors="replace")
    
    def __len__(self):
        return len(self.tokenizer)