import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset

class FineWeb(IterableDataset):
    def __init__(self, tokenizer, context_size, device='cpu'):
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.device = device
       
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu", 
            name='sample-10BT', 
            trust_remote_code=True, 
            split='train',
            streaming=True
        ).shuffle(buffer_size=10_000, seed=411)

    def __iter__(self):

        for data in self.dataset:
            input_ids = self.tokenizer.encode(data['text']).squeeze(0)

            chunks = torch.split(input_ids, self.context_size + 1)
            
            for chunk in chunks:
                if chunk.size(0) < self.context_size + 1:
                    pad_size = self.context_size + 1 - chunk.size(0)
                    pad = torch.full((pad_size,), self.tokenizer.pad_token_id, dtype=torch.long)
                    chunk = torch.cat([chunk, pad])
                
                yield chunk[:-1].to(self.device), chunk[1:].to(self.device)