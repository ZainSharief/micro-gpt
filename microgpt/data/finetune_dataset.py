import torch
from datasets import load_dataset
from torch.utils.data import Dataset

class HH_RLHF_Chosen(Dataset):

    def __init__(self, tokenizer, context_size=384, split='train', device='cpu'):

        self.data = load_dataset(
            'Anthropic/hh-rlhf', 
            split=split, 
            trust_remote_code=True
        )['chosen']

        self.context_size = context_size
        self.tokenizer = tokenizer
        self.device = device
    
    def __len__(self):
        return len(self.data)
    
    def build_loss_mask(self, y, assistant_id, end_assistant_id):
       
        loss_mask = torch.zeros_like(y)

        # find all positions
        starts = (y == assistant_id).nonzero(as_tuple=True)[0]
        ends = (y == end_assistant_id).nonzero(as_tuple=True)[0]

        # pair them safely
        for s in starts:
            after = ends[ends > s]
            if len(after) > 0:
                e = after[0].item()
                loss_mask[s+1:e+1] = 1

        return loss_mask

    def __getitem__(self, idx):

        data = self.data[idx].strip().replace('\n', '')
        data = data.replace('Human: ', self.tokenizer.end_assistant_token + self.tokenizer.user_token)
        data = data.replace('Assistant: ', self.tokenizer.end_user_token + self.tokenizer.assistant_token)

        # removes the unneccessary end_assistant_token at the start
        data = data[len(self.tokenizer.end_assistant_token):]
        data += self.tokenizer.end_assistant_token

        data = self.tokenizer.encode_padding(data)
        x = data[:-1]
        y = data[1:]

        loss_mask = self.build_loss_mask(y, self.tokenizer.assistant_token_id, self.tokenizer.end_assistant_token_id)

        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True), loss_mask.to(self.device, non_blocking=True)