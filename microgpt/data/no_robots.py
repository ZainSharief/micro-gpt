import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from microgpt.tokenizer import GPTtokenizer

class NoRobots(Dataset):
    """https://huggingface.co/datasets/HuggingFaceH4/no_robots"""

    def __init__(
            self, 
            tokenizer: GPTtokenizer,
            context_size: int = 384,
            split: str = 'train',
            device: str = 'cpu'
        ):

        self.data = load_dataset(
            'HuggingFaceH4/no_robots', 
            split=split, 
            trust_remote_code=True
        )

        # our model is not big enough to learn facts, so we exclude those
        categories = ['Summarize', 'Rewrite', 'Classify', 'Extract', 'Closed QA']
        self.data = [x['messages'] for x in self.data if x['category'] in categories]

        self.context_size = context_size
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        conversation = self.data[idx]
        text = ""

        for message in conversation:
            text += self.tokenizer.user_token if message['role'] == 'user' else self.tokenizer.assistant_token
            text += message['content']
            text += self.tokenizer.end_user_token if message['role'] == 'user' else self.tokenizer.end_assistant_token

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
    dataset = NoRobots(tokenizer, context_size=config.context_size)
    x, y, loss = dataset.__getitem__(1)
    print(x, y, loss.sum())