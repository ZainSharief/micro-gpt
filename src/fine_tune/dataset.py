import torch
from datasets import load_dataset

class OpenAssistant(torch.utils.data.Dataset):
    def __init__(self, tokeniser, context_size, batch_size, device, pad_token_id=0):
        self.tokeniser = tokeniser
        self.context_size = context_size
        self.batch_size = batch_size
        self.device = device
        self.pad_token_id = pad_token_id

        self.data = load_dataset("OpenAssistant/oasst2", trust_remote_code=True, split='train')
        self.data = self.data.filter(lambda x: x['lang'] == 'en')
        self.data = self.format_conversations(self.data)
        self.__getitem__(0)

    def format_conversations(self, dataset):
        message_dict = {entry["message_id"]: entry for entry in dataset}
        messages = []

        for entry in dataset:
            if entry["role"] != "prompter":
                continue
            response_entry = message_dict.get(entry.get("parent_id"))
            if response_entry and response_entry["role"] == "assistant":
                messages.append({"prompt": entry["text"], "response": response_entry["text"]})

        return messages
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
                
        prompt_tokens = self.tokeniser.encode(text['prompt'])
        response_tokens = self.tokeniser.encode(text['response'])
        prompt_length = prompt_tokens.size()[-1]
        og = prompt_length
        response_length = response_tokens.size()[-1]

        if prompt_length + response_length + 1 < self.context_size:

            x = torch.cat([
                prompt_tokens.to(self.device), 
                torch.tensor(self.tokeniser.eos_token, dtype=torch.long, device=self.device).reshape([1, 1]), 
                response_tokens.to(self.device), 
                torch.zeros([1, self.context_size - (prompt_length + response_length + 1)], device=self.device, dtype=torch.long)
            ], dim=-1)

            y = torch.cat([
                torch.zeros([1, prompt_length], dtype=torch.long, device=self.device),
                response_tokens.to(self.device),
                torch.zeros([1, self.context_size - (prompt_length + response_length)], device=self.device, dtype=torch.long)
            ], dim=-1)

            x, y = x.to(self.device), y.to(self.device)
            return x.squeeze(0), y.squeeze(0)

        if prompt_length > self.context_size:
            prompt_tokens = prompt_tokens[:, -self.context_size+1:]
            prompt_length = self.context_size - 1 

        x = torch.cat([
            prompt_tokens.to(self.device), 
            torch.tensor(self.tokeniser.eos_token, dtype=torch.long, device=self.device).reshape([1, 1]), 
            response_tokens.to(self.device)
        ], dim=-1)

        y = torch.cat([
            torch.zeros([1, prompt_length+1], dtype=torch.long, device=self.device),
            response_tokens.to(self.device)
        ], dim=-1)

        if x.size()[-1] > self.context_size:
            start = torch.randint(0, x.size()[-1]-self.context_size, (1,), device=self.device)
            x = x[:, start:start+self.context_size]
            y = y[:, start:start+self.context_size]

        x, y = x.to(self.device), y.to(self.device)
        return x.squeeze(0), y.squeeze(0)