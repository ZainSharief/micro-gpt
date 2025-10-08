from transformers import GPT2Tokenizer
from microgpt.config import Config
import torch

class GPTtokenizer():

    """
    Wrapper around the HuggingFace GPT2 tokenizer with added special tokens for user and assistant.

    Special Tokens:
    - <|user|>: Token indicating the start of a user message.
    - <|enduser|>: Token indicating the end of a user message.
    - <|assistant|>: Token indicating the start of an assistant message.
    - <|endassistant|>: Token indicating the end of an assistant message.
    """

    def __init__(self):

        """
        Initializes the GPTtokenizer by loading the GPT2 tokenizer and adding special tokens.
        Also sets up various attributes for easy access to token IDs and vocabulary size.
        """

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=True, verbose=False, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.decode(Config().pad_token_id)
        self.tokenizer.truncation_side = "left"
    
        new_tokens = {
            'user_token': '<|user|>',
            'end_user_token': '<|enduser|>',
            'assistant_token': '<|assistant|>',
            'end_assistant_token': '<|endassistant|>'
        }
        self.add_tokens(new_tokens)
        
        self.vocab_size = self.tokenizer.vocab_size

        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        
    def add_tokens(self, tokens):

        # adds tokens to the tokenizer and creates attributes for easy access
        self.tokenizer.add_tokens(list(tokens.values()))
        for name, token in tokens.items():
            setattr(self, name, token)

        for name, token in tokens.items():
            setattr(self, f'{name}_id', int(self.tokenizer.convert_tokens_to_ids(token)))
    
    def encode(self, data: str) -> torch.Tensor:

        """
        Encodes the input text data into token IDs.

        Args:
            data (str): The input text to be tokenized.

        Returns:
            torch.Tensor: A tensor of token IDs with shape (1, sequence_length).

        """

        return self.tokenizer(data, return_tensors="pt")["input_ids"]
    
    def encode_padding(self, data: str, max_length: int = 384) -> torch.Tensor:

        """
        Encodes the input text data into token IDs with padding to a specified maximum length.

        Args:
            data (str): The input text to be tokenized.
            max_length (int): The maximum length for padding/truncation.

        Returns:
            torch.Tensor: A tensor of token IDs with shape (max_length,).
        """

        return self.tokenizer(data, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")["input_ids"].squeeze(0)
 
    def decode(self, tokens: torch.Tensor) -> str:

        """
        Decodes a tensor of token IDs back into a string.

        Args:
            tokens (torch.Tensor): A tensor of token IDs.

        Returns:
            str: The decoded string.
        """

        return self.tokenizer.decode(tokens, errors="replace")
    
    def __len__(self):
        return len(self.tokenizer)