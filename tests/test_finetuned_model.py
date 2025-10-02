import torch
from torch.utils.data import DataLoader
import time

from microgpt.tokenizer import GPTtokenizer
from microgpt.config import Config
from microgpt.model.model import FinetuneModel

'''
Testing the architecture of the base model.

If the base model is working, it should successfully overfit the dataset
and the loss will converge to 0.
'''

class DummyDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer: GPTtokenizer, context_size: int = 384, device: str = 'cpu'):

        self.tokenizer = tokenizer
        self.context_size = context_size
        self.device = device

        text = self.tokenizer.user_token + 'What is the captial of France?' + self.tokenizer.end_user_token \
        + self.tokenizer.assistant_token + 'The capital of France is Paris.' + self.tokenizer.end_assistant_token

        self.data = self.tokenizer.encode(text).to(device).squeeze(0)
        self.loss_mask = self.build_loss_mask(self.data[1:], self.tokenizer.assistant_token, self.tokenizer.end_assistant_token)

        if self.data.size(-1) > self.context_size:
            self.data = self.data[:self.context_size]
        
        elif self.data.size(-1) < self.context_size:
            pad_size = self.context_size - self.data.size(-1)
            self.data = torch.concat([self.data, torch.zeros(pad_size + 1, dtype=torch.long, device=device)], dim=-1)
    
    def build_loss_mask(self, y, assistant_id, end_assistant_id):
       
        loss_mask = torch.zeros_like(y)

        starts = (y == assistant_id).nonzero(as_tuple=True)[0]
        ends = (y == end_assistant_id).nonzero(as_tuple=True)[0]

        for s in starts:
            after = ends[ends > s]
            if len(after) > 0:
                e = after[0].item()
                loss_mask[s+1:e+1] = 1

        return loss_mask

    def __len__(self):
        return 1_000_000

    def __getitem__(self, _):  
        return self.data[:-1], self.data[1:], self.loss_mask
    
    def get_first_token(self):
        return self.tokenizer.decode(self.data[0])

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # dummy trainig params
    batch_size = 32                      
    learning_rate = 4e-4
    max_lr = 8e-4

    config = Config()
    tokenizer = GPTtokenizer()
    dataset = DummyDataset(tokenizer=tokenizer, context_size=config.context_size, device=device) 
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model = FinetuneModel(config, use_lora=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_lr,
        total_steps=1_000_000,
        pct_start=0.1, 
        anneal_strategy='cos'
    )
    scaler = torch.amp.GradScaler(device)
    model = model.to(device)
    
    for current_batch, (xb, yb, loss_mask) in enumerate(dataloader):

        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device, dtype=torch.float16):
            _, loss = model(xb, yb, loss_mask)
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_time = time.time() - start_time
    
        print(f"\rbatch: {current_batch+1}/{len(dataset)} | loss: {loss:.4f} | lr: {scheduler.get_last_lr()[0]:.4e} | step_time: {int(total_time*1000)}ms", end='') 
        
if __name__ == '__main__':
    main()