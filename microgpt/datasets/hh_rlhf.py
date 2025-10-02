import torch
from datasets import load_dataset
from torch.utils.data import Dataset
import os
import re

from microgpt.tokenizer import GPTtokenizer

class HH_RLHF(Dataset):

    """
    Designed for a reward model. Each sample consists of a pair of input-output sequences (chosen and rejected),
    along with a dummy mask for training compatibility.
    
    Credits:
        HH_RLHF dataset from HuggingFace: https://huggingface.co/datasets/Anthropic/hh-rlhf
    """

    def __init__(
            self, 
            tokenizer: GPTtokenizer, 
            context_size: int = 384, 
            save_path: str = "hh_rlhf_tokens.bin", 
            split: str = 'train', 
            device: str = 'cpu'
        ):

        """
        Initializes the HH_RLHF dataset. If the pre-processed token file exists at save_path, it is loaded.
        Otherwise, it preprocesses the dataset and saves it to save_path.

        Args:
            tokenizer (GPTtokenizer): The tokenizer to use for encoding text.
            context_size (int): The size of the context window for each sample.
            save_path (str): The path to save/load the pre-processed token file.
            split (str): The dataset split to use ('train' or 'test').
            device (str): The device to load the data onto ('cpu' or 'cuda').
        """

        self.context_size = context_size
        self.tokenizer = tokenizer
        self.device = device

        if not os.path.exists(save_path):
            self.preprocess_and_save(split, save_path)

        self.data = torch.load(save_path)
        self.dummy_mask = torch.tensor(0)

    def split_conversation(self, conversation: str) -> list[str]:

        # Splits the conversation into a list of human and assistant texts
        pattern = r"(Human|Assistant):\s*"
        matches = list(re.finditer(pattern, conversation))

        last_role = None
        merged_texts = []

        # For some reason, the dataset has some consecutive messages from the same role
        # so we merge them together
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
        
        # Remove any empty strings from the list
        merged_texts = [t for t in merged_texts if t]

        return merged_texts

    def preprocess_and_save(self, split: str, save_path: str) -> None:

        dataset = load_dataset(
            'Anthropic/hh-rlhf', 
            split=split, 
            trust_remote_code=True
        )

        samples = []
        for item in dataset:

            chosen = self.split_conversation(item['chosen'])
            rejected = self.split_conversation(item['rejected'])

            # ensures that the number of human and assistant texts are the same
            if (len(chosen) != len(rejected)) or len(chosen) % 2:
                continue

            for i in range(0, len(chosen), 2):

                # reject samples where the assistant responses are the same
                if chosen[i+1] == rejected[i+1]:
                    continue

                # surrounds user and assistant texts with special tokens
                chosen_sample = self.tokenizer.encode_padding(f'{self.tokenizer.user_token}{chosen[i]}{self.tokenizer.end_user_token}{self.tokenizer.assistant_token}{chosen[i+1]}{self.tokenizer.end_assistant_token}', max_length=self.context_size)
                rejected_sample = self.tokenizer.encode_padding(f'{self.tokenizer.user_token}{chosen[i]}{self.tokenizer.end_user_token}{self.tokenizer.assistant_token}{rejected[i+1]}{self.tokenizer.end_assistant_token}', max_length=self.context_size)
                samples.append((chosen_sample, rejected_sample))
        
        torch.save(samples, save_path)
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        """
        Retrieves a sample from the dataset at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            x (torch.Tensor): The input tensor of shape (context_size,).
            y (torch.Tensor): The target tensor of shape (context_size,).
            loss_mask (torch.Tensor): The loss mask tensor of shape (context_size,).
        """

        return *self.data[idx], self.dummy_mask