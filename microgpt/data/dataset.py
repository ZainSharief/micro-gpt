import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset
import numpy as np
import os
import re

from microgpt.tokenizer import GPTtokenizer

class FineWeb(Dataset):
    def __init__(self, tokenizer, context_size, save_path="fineweb_tokens.bin", device='cpu'):
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.save_path = save_path
        self.device = device

        if not os.path.exists(save_path):
            self.preprocess_and_save()

        self.data = np.memmap(save_path, dtype=np.int64, mode="r")
        self.dummy_mask = torch.tensor(0)

    def __len__(self):
        return len(self.data) // (self.context_size + 1)
    
    def __getitem__(self, idx):

        i = idx * (self.context_size + 1)
        xy = self.data[i:i + (self.context_size + 1)]

        x = torch.tensor(xy[:-1], dtype=torch.long)
        y = torch.tensor(xy[1:], dtype=torch.long)

        return x, y, self.dummy_mask
     
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

        data = self.tokenizer.encode_padding(data, max_length=self.context_size+1)
        x = data[:-1]
        y = data[1:]

        loss_mask = self.build_loss_mask(y, self.tokenizer.assistant_token_id, self.tokenizer.end_assistant_token_id)

        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True), loss_mask.to(self.device, non_blocking=True)
    
class HH_RLHF(Dataset):

    def __init__(self, tokenizer: GPTtokenizer, context_size: int = 384, save_path: str = "hh_rlhf_tokens.bin", split: str = 'train', device: str = 'cpu'):
        self.context_size = context_size
        self.tokenizer = tokenizer
        self.device = device

        if not os.path.exists(save_path):
            self.preprocess_and_save(split, save_path)

        self.data = torch.load(save_path)
        self.dummy_mask = torch.tensor(0)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return *self.data[idx], self.dummy_mask
    
    def split_conversation(self, conversation):

        pattern = r"(Human|Assistant):\s*"
        matches = list(re.finditer(pattern, conversation))

        last_role = None
        merged_texts = []

        for i in range(len(matches)):
            
            role = matches[i].group(1)
            start = matches[i].end()
            end = matches[i+1].start() if i+1 < len(matches) else len(conversation)
            text = conversation[start:end].strip()
            
            if last_role == role:
                merged_texts[-1] += "\n\n" + text
            else:
                merged_texts.append(text)
                last_role = role
        
        merged_texts = [t for t in merged_texts if t]

        return merged_texts

    def preprocess_and_save(self, split, save_path):

        dataset = load_dataset(
            'Anthropic/hh-rlhf', 
            split=split, 
            trust_remote_code=True
        )

        samples = []
        total = 0
        for item in dataset:

            chosen = self.split_conversation(item['chosen'])
            rejected = self.split_conversation(item['rejected'])

            if (len(chosen) != len(rejected)) or len(chosen) % 2:
                continue

            for i in range(0, len(chosen), 2):

                if chosen[i+1] == rejected[i+1]:
                    continue

                chosen_sample = self.tokenizer.encode_padding(f'{self.tokenizer.user_token}{chosen[i]}{self.tokenizer.end_user_token}{self.tokenizer.assistant_token}{chosen[i+1]}{self.tokenizer.end_assistant_token}', max_length=self.context_size)
                rejected_sample = self.tokenizer.encode_padding(f'{self.tokenizer.user_token}{chosen[i]}{self.tokenizer.end_user_token}{self.tokenizer.assistant_token}{rejected[i+1]}{self.tokenizer.end_assistant_token}', max_length=self.context_size)
                samples.append((chosen_sample, rejected_sample))
                total += 1
        
        torch.save(samples, save_path)