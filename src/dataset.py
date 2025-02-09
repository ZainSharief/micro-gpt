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

    def __getbatch__(self, idx):
        text = self.tokeniser.eos_token + self.data[idx]['text'] + self.tokeniser.eos_token
        tokens = self.tokeniser.encode(text)

        x = tokens[:, :-1]
        y = tokens[:, 1:]

        B, T = x.size()
        batch_size = min(self.batch_size, T//self.context_size)

        if batch_size > 0:
            x = x[:, :batch_size*self.context_size].view(batch_size, self.context_size)
            y = y[:, :batch_size*self.context_size].view(batch_size, self.context_size)

        x, y = x.to(self.device), y.to(self.device)
        return x, y  