import torch
from datasets import load_dataset
from torch.utils.data import Dataset
import numpy as np
import os
import re

from microgpt.tokenizer import GPTtokenizer

class FineWeb(Dataset):

    """
    Designed for pre-training. Each sample is a sequence of token IDs representing contiguous chunks of text, 
    with a dummy mask included for training compatibility.

    Credits:
        FineWeb dataset from HuggingFace: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
    """

    def __init__(
            self, 
            tokenizer: GPTtokenizer,
            context_size : int = 384,
            save_path: str = "fineweb_tokens.bin",
            device: str = 'cpu'
        ):

        """
        Initializes the FineWeb dataset. If the pre-processed token file exists at save_path, it is loaded.
        Otherwise, it preprocesses the dataset and saves it to save_path.

        Args:
            tokenizer (GPTtokenizer): The tokenizer to use for encoding text.
            context_size (int): The size of the context window for each sample.
            save_path (str): The path to save/load the pre-processed token file.
            device (str): The device to load the data onto ('cpu' or 'cuda').
        """

        self.tokenizer = tokenizer
        self.context_size = context_size
        self.save_path = save_path
        self.device = device

        if not os.path.exists(save_path):
            self.preprocess_and_save()

        self.data = np.memmap(save_path, dtype=np.int64, mode="r")
        self.dummy_mask = torch.tensor(0)

    def preprocess_and_save(self) -> None:

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

    def __len__(self) -> int:
        return len(self.data) // (self.context_size + 1)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        """
        Retrieves a sample from the dataset at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            x (torch.Tensor): The input tensor of shape (context_size,).
            y (torch.Tensor): The target tensor of shape (context_size,).
            dummy_mask (torch.Tensor): A dummy tensor (always 0) for dataset compatability.
        """

        i = idx * (self.context_size + 1)
        xy = self.data[i:i + (self.context_size + 1)]

        x = torch.tensor(xy[:-1], dtype=torch.long)
        y = torch.tensor(xy[1:], dtype=torch.long)

        return x, y, self.dummy_mask

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