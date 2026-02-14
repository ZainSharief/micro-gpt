import torch
from datasets import load_dataset
from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm
from microgpt.tokenizer.tokenizer import GPTtokenizer

class FineWeb(Dataset):

    def __init__(
            self, 
            tokenizer: GPTtokenizer,
            context_size : int,
            save_path: str = "fineweb_tokens.bin",
            device: str = 'cpu'
        ):

        self.tokenizer = tokenizer
        self.context_size = context_size
        self.save_path = save_path
        self.device = device

        if not os.path.exists(save_path):
            self.preprocess_and_save()

        self.data = np.memmap(save_path, dtype=np.uint16, mode="r")
        self.dummy_mask = torch.tensor(0)

    def preprocess_and_save(self) -> None:
        
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu", 
            name='sample-10BT', 
            trust_remote_code=True, 
            split='train',
            streaming=True 
        )

        batch_texts = []
        batch_size = 50_000 # change based on RAM

        with open(self.save_path, "wb") as f:

            for _, doc in tqdm(enumerate(dataset), desc="Tokenizing"):
                text = doc['text'] + self.tokenizer.eos_token
                batch_texts.append(text)

                if len(batch_texts) >= batch_size:
                    self.flush_batch(batch_texts, f)
                    batch_texts = []

            if batch_texts:
                self.flush_batch(batch_texts, f)
        
    def flush_batch(self, batch_texts, f):
        """Helper to tokenize a batch and write bytes to file handle."""
        encoded_batch = self.tokenizer.tokenizer(
            batch_texts, 
            padding=False, 
            truncation=False
        )['input_ids']

        flat_tokens = [token for seq in encoded_batch for token in seq]
        arr = np.array(flat_tokens, dtype=np.uint16)
        arr.tofile(f)

    def __len__(self) -> int:
        return len(self.data) // (self.context_size + 1)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = idx * (self.context_size + 1)
        xy = self.data[i:i + (self.context_size + 1)]

        x = torch.tensor(xy[:-1], dtype=torch.long)
        y = torch.tensor(xy[1:], dtype=torch.long)

        return x, y, self.dummy_mask
    
if __name__ == '__main__':
    from microgpt import Config
    config = Config()

    tokenizer = GPTtokenizer()
    dataset = FineWeb(tokenizer, context_size=config.context_size)