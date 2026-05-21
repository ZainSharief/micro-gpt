import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from microgpt.tokenizer import GPTtokenizer
import numpy as np
import os

class Oasst1(Dataset):
    """https://huggingface.co/datasets/OpenAssistant/oasst1"""
    
    def __init__(
        self, 
        tokenizer: GPTtokenizer,
        context_size: int = 1024,
        split: str = 'train',
        device: str = 'cpu'
    ):
        
        data = load_dataset(
            'OpenAssistant/oasst1', 
            split=split, 
            trust_remote_code=True
        )

        self.save_path = 'oasst1_tokens.bin'
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.device = device
        self.data = self.generate_dataset(data)
        self.loss_mask = self.build_loss_mask(self.data)
        
    def __len__(self):
        return len(self.data) // (self.context_size + 1)
    
    def __getitem__(self, idx):
        start = idx * (self.context_size + 1)
        end = start + self.context_size + 1

        x = self.data[start:end-1].to(torch.long)
        y = self.data[start+1:end].to(torch.long)
        loss_mask = self.loss_mask[start+1:end]

        return x.to(self.device), y.to(self.device), loss_mask.to(self.device)
    
    def generate_dataset(self, data) -> torch.Tensor:

        if os.path.exists(self.save_path):
            return torch.from_numpy(np.memmap(self.save_path, dtype=np.uint16, mode="r"))

        non_leaves = set(msg['parent_id'] for msg in data)
        parents = {msg['message_id']: msg for msg in data}

        conversations = []
        for i, msg in enumerate(data):

            if msg['message_id'] in non_leaves or msg['lang'] != 'en':
                continue

            contains_assistant = msg['role'] == 'assistant'
            start_tok = self.tokenizer.user_token if msg['role'] == 'prompter' else self.tokenizer.assistant_token
            end_tok = self.tokenizer.end_user_token if msg['role'] == 'prompter' else self.tokenizer.end_assistant_token
            current = [start_tok + msg['text'] + end_tok]
            parent = parents.get(msg['parent_id'], None)
            while parent:
                start_tok = self.tokenizer.user_token if parent['role'] == 'prompter' else self.tokenizer.assistant_token
                end_tok = self.tokenizer.end_user_token if parent['role'] == 'prompter' else self.tokenizer.end_assistant_token
                current.append(start_tok + parent['text'] + end_tok)
                parent = parents.get(parent['parent_id'], None)

                if msg['role'] == 'assistant':
                    contains_assistant = True

                if msg['lang'] != 'en':
                    continue

            if not contains_assistant:
                continue

            current.reverse()
            text = ''.join(current)
            tokens = self.tokenizer.encode(text)[0]
            conversations.append(tokens)
            conversations.append(torch.tensor([self.tokenizer.eos_token_id]))

            print(f'\rcompleted: {i+1}/{len(data)}', end='')

        dataset = torch.cat(conversations, dim=0).flatten()
        save_dataset = np.array(dataset, dtype=np.uint16)
        save_dataset.tofile(self.save_path)
        return dataset

    def build_loss_mask(self, data_tensor: torch.Tensor) -> torch.Tensor:
        loss_mask = torch.zeros_like(data_tensor, dtype=torch.float)

        assistant_starts = (data_tensor == self.tokenizer.assistant_token_id).nonzero(as_tuple=True)[0]
        assistant_ends = (data_tensor == self.tokenizer.end_assistant_token_id).nonzero(as_tuple=True)[0]

        for start in assistant_starts:
            valid_ends = assistant_ends[assistant_ends > start]
            
            if len(valid_ends) > 0:
                end = valid_ends[0]
            else:
                end = len(data_tensor) - 1 

            if end >= start:
                loss_mask[start + 1 : end + 1] = 1.0
                
        return loss_mask
    
if __name__ == '__main__':
    from microgpt import Config
    config = Config()

    torch.set_printoptions(profile='full')
    tokenizer = GPTtokenizer()
    dataset = Oasst1(tokenizer, context_size=config.context_size, split='train')
    print(len(dataset))
    print(dataset.__getitem__(0))