import torch
from datasets import load_dataset
from torch.utils.data import Dataset
import numpy as np
import os

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