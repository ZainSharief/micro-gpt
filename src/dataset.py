import torch
from datasets import load_dataset

class FineWeb(torch.utils.data.Dataset):
    def __init__(self, tokeniser, context_size, batch_size, device):
        self.data = load_dataset("HuggingFaceFW/fineweb-edu", name='sample-10BT', trust_remote_code=True, split='train')
        self.tokeniser = tokeniser
        self.context_size = context_size
        self.batch_size = batch_size
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        tokens = self.tokeniser.encode(text)

        x = tokens[:, :-1]
        y = tokens[:, 1:]

        B, T = x.size()

        if T <= self.context_size:
            pad = torch.zeros((B, self.context_size - T), device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
            y = torch.cat([y, pad], dim=1)
        else:
            start = torch.randint(0, T - self.context_size, (1,), device=x.device)

            x = x[:, start:start+self.context_size]
            y = y[:, start:start+self.context_size]

        x, y = x.to(self.device), y.to(self.device)
        return x.squeeze(0), y.squeeze(0)