import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from microgpt.tokenizer import GPTtokenizer

class Alpaca(Dataset):
    
    def __init__(
        self, 
        tokenizer: GPTtokenizer,
        context_size: int = 384,
        split: str = 'train',
        device: str = 'cpu'
    ):

        self.data = load_dataset(
            'tatsu-lab/alpaca', 
            split=split, 
            trust_remote_code=True
        )

        self.context_size = context_size
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # alpaca has 'instruction', 'input' (optional context), and 'output'
        user_text = f"{item['instruction']}\n\n{item['input']}" if item['input'] else item['instruction']
        assistant_text = item['output']

        # Format: <|user|> instruction <|enduser|> <|assistant|> response <|endassistant|>
        text = (
            self.tokenizer.user_token + 
            user_text + 
            self.tokenizer.end_user_token +
            self.tokenizer.assistant_token + 
            assistant_text + 
            self.tokenizer.end_assistant_token
        )

        tokens = self.tokenizer.encode_padding(text, max_length=self.context_size + 1)
        x = tokens[:-1]
        y = tokens[1:]
        loss_mask = self.build_loss_mask(y)
        
        return x.to(self.device), y.to(self.device), loss_mask.to(self.device)
    
    def build_loss_mask(self, y: torch.Tensor) -> torch.Tensor:

        loss_mask = torch.zeros_like(y)
        assistant_starts = (y == self.tokenizer.assistant_token_id).nonzero(as_tuple=True)[0]
        assistant_ends = (y == self.tokenizer.end_assistant_token_id).nonzero(as_tuple=True)[0]

        min_len = min(len(assistant_starts), len(assistant_ends))
        
        for i in range(min_len):
            start = assistant_starts[i]
            end = assistant_ends[i]
            
            if end > start:
                loss_mask[start + 1 : end + 1] = 1.0
                
        return loss_mask
    
if __name__ == '__main__':
    from microgpt import Config
    config = Config()

    tokenizer = GPTtokenizer()
    dataset = Alpaca(tokenizer, context_size=config.context_size)
    x, y, loss = dataset.__getitem__(1)
    print(len(dataset))
    print(x, y, loss.sum())
