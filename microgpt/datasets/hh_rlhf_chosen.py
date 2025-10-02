import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from microgpt.tokenizer import GPTtokenizer

class HH_RLHF_Chosen(Dataset):

    """
    Designed for supervised fine-tuning. Each sample consists of input-output pairs where the model is 
    trained to generate the assistant's response given the user's input. A loss mask is provided to ensure 
    that only the assistant's response is considered during training.

    Credits:
        HH_RLHF dataset from HuggingFace: https://huggingface.co/datasets/Anthropic/hh-rlhf
    """

    def __init__(
            self, 
            tokenizer: GPTtokenizer,
            context_size: int = 384,
            split: str = 'train',
            device: str = 'cpu'
        ):

        """
        Initializes the HH_RLHF_Chosen dataset.

        Args:
            tokenizer (GPTtokenizer): The tokenizer to use for encoding text.
            context_size (int): The size of the context window for each sample.
            split (str): The dataset split to use ('train' or 'validation').
            device (str): The device to load the data onto ('cpu' or 'cuda').
        """

        self.data = load_dataset(
            'Anthropic/hh-rlhf', 
            split=split, 
            trust_remote_code=True
        )['chosen']

        self.context_size = context_size
        self.tokenizer = tokenizer
        self.device = device
    
    def __len__(self) -> int:
        return len(self.data)
    
    def build_loss_mask(self, y: torch.Tensor, assistant_id: int, end_assistant_id: int) -> torch.Tensor:
        
        """
        Builds a loss mask for the target tensor y.
        Given the token structure: user_start uster_text user_end assistant_start assistant_text assistant_end
        The loss mask will will block user_start to assistant_start, so the model only predicts assistant_text + assistant_end.

        Args:
            y (torch.Tensor): The target tensor of shape (context_size,).
            assistant_id (int): The token ID representing the start of the assistant's response.
            end_assistant_id (int): The token ID representing the end of the assistant's response.

        Returns:
            loss_mask (torch.Tensor): The loss mask tensor of shape (context_size,).
        """

        loss_mask = torch.zeros_like(y)

        starts = (y == assistant_id).nonzero(as_tuple=True)[0]
        ends = (y == end_assistant_id).nonzero(as_tuple=True)[0]

        for s in starts:
            after = ends[ends > s]
            if len(after) > 0:
                e = after[0].item()
                loss_mask[s+1:e+1] = 1

        return loss_mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        """
        Preprocesses and Retrieves a sample from the dataset at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            x (torch.Tensor): The input tensor of shape (context_size,).
            y (torch.Tensor): The target tensor of shape (context_size,).
            loss_mask (torch.Tensor): The loss mask tensor of shape (context_size,).
        """

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