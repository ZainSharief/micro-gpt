import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from microgpt.tokenizer import GPTtokenizer
import re

class DailyDialog(Dataset):
    """https://huggingface.co/datasets/roskoN/dailydialog"""
    
    def __init__(
        self, 
        tokenizer: GPTtokenizer,
        context_size: int = 1024,
        split: str = 'train',
        device: str = 'cpu'
    ):
        
        data = load_dataset(
            'roskoN/dailydialog', 
            split=split, 
            trust_remote_code=True
        )

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

        x = self.data[start:end-1]
        y = self.data[start+1:end]
        loss_mask = self.loss_mask[start+1:end]

        return x.to(self.device), y.to(self.device), loss_mask.to(self.device)
    
    def generate_dataset(self, data) -> torch.Tensor:
        """Combine and tokensize the conversations using sequence packing """
        dataset = []
        tok = self.tokenizer
        for item in data:
            text = item['utterances']
            conversation = ""
            for i, utterance in enumerate(text):
                start_token, end_token = (tok.user_token, tok.end_user_token) if i % 2 == 0 else (tok.assistant_token, tok.end_assistant_token)
                cleaned = self.clean_dailydialog_text(utterance)
                conversation += f'{start_token}{cleaned}{end_token}'

            dataset.append(conversation)

        dataset = str(self.tokenizer.eos_token).join(dataset)
        tokens = self.tokenizer.encode(dataset).flatten()
        tokens = tokens[:(len(tokens) // (self.context_size+1)) * (self.context_size+1)] # truncate to prevent overflow text 
        return tokens

    def clean_dailydialog_text(self, text: str) -> str:
        text = re.sub(r'\s+([?.!,:;])', r'\1', text) # remove spaces before punctuation
        text = re.sub(r"(\w)\s+'\s+(\w)", r"\1'\2", text) # fix separated contractions (e.g., "I ' m" -> "I'm", "don ' t" -> "don't")
        text = re.sub(r"\s+'\s+", "'", text) # fix lone apostrophes at the start or end of words
        text = re.sub(r'\s+', ' ', text) # clean up double spaces created by the removals
        return text.strip()
    
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
    dataset = DailyDialog(tokenizer, context_size=config.context_size, split='train')
    print(len(dataset))
    print(dataset.__getitem__(0))