import torch
from datasets import load_dataset

class UnknownDataset(torch.utils.data.Dataset):
    def __init__(self, tokeniser, context_size, batch_size, device, pad_token_id=0):
        raise NotImplementedError()

    