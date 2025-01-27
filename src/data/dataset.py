import torch
import torch.nn.functional as F

from datasets import load_dataset
from re import sub

class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokeniser, context_size, batch_size, device, train=True):
        self.data = load_dataset("daily_dialog", trust_remote_code=True)['train' if train else 'validation']
        self.tokeniser = tokeniser
        self.context_size = context_size
        self.batch_size = batch_size
        self.device = device

    def __len__(self):
        return len(self.data)

    def preprocess_text(self, conversation):
        text = ''
        for line in conversation:

            line = line.strip()
            line = sub(r'\s([,.?!])', r'\1', line)
            line = sub(r"(\b\w+)\s'\s(\w+)", r"\1'\2", line)
            line = sub(r" ’ ", r"'", line)

            text += line + self.tokeniser.eos_token 

        text = text[:-len(self.tokeniser.eos_token)] 
        return text

    def __getitem__(self, idx):
        conversation = self.data[idx]['dialog']
        text = self.preprocess_text(conversation)
        data = self.tokeniser.encode(text).view(-1)

        data = data[:self.context_size+1]
        x = data[:-1].unsqueeze(0)
        y = data[1:].unsqueeze(0)

        x, y = x.to(self.device), y.to(self.device)
        return x, y  
    
    def __getbatch__(self, idx):
        items = [self[x] for x in range(idx, idx + self.batch_size)]
        xbatch, ybatch = zip(*items)
        
        max_size = max(item.size(-1) for item in xbatch)
        
        xbatch = torch.cat([F.pad(item, (0, max_size-item.size(-1))) for item in xbatch], dim=0)
        ybatch = torch.cat([F.pad(item, (0, max_size-item.size(-1))) for item in ybatch], dim=0)
        
        return xbatch, ybatch


    def get_text(self):
        text = ''
        for conversation in self.data:
            conversation = conversation['dialog']
            text += self.preprocess_text(conversation)
        
        return text
    
if __name__ == '__main__':
    from GPT2tokeniser import GPTtokenizer

    tokeniser = GPTtokenizer()
    dataset = Dataset(tokeniser, 256, 32, 'cpu')
    for i in range(len(dataset) // 32):
        x, y = dataset.__getbatch__(i)
        print(x[0], y[0])
        quit(0)