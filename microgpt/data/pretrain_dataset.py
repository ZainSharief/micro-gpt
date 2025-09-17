import torch
from datasets import load_dataset
from torch.utils.data import Dataset
import numpy as np
import os

from microgpt.tokenizer import GPTtokenizer

class FineWeb(Dataset):
    def __init__(self, tokenizer : GPTtokenizer, context_size : int, save_path : str = "fineweb_tokens.bin", device : str = 'cpu'):
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.save_path = save_path
        self.device = device

        if not os.path.exists(save_path):
            self.preprocess_and_save()

        self.data = np.memmap(save_path, dtype=np.int64, mode="r")

    def __len__(self):
        return len(self.data) // (self.context_size + 1)
    
    def __getitem__(self, idx):

        i = idx * (self.context_size + 1)
        xy = self.data[i:i + (self.context_size + 1)]

        x = torch.tensor(xy[:-1], dtype=torch.long)
        y = torch.tensor(xy[1:], dtype=torch.long)

        return x, y
     
    def preprocess_and_save(self):

        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu", 
            name='sample-10BT', 
            trust_remote_code=True, 
            split='train'
        )

        with open(self.save_path, "wb") as fout:
            batch_texts = [doc['text'] + self.tokenizer.eos_token for doc in dataset]
            all_tokens = self.tokenizer.tokenizer(batch_texts, return_tensors='np', padding=False, truncation=False)['input_ids']
            np.concatenate(all_tokens).tofile(fout)
