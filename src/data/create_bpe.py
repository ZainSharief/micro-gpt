from datasets import load_dataset
from preprocess import get_text

from BPEtokeniser import BPETokeniser

if __name__ == '__main__':

    tokeniser_path = 'byte-pair-encoding10000.pkl'

    dataset = load_dataset("daily_dialog", trust_remote_code=True)
    tokeniser = BPETokeniser(num_merges=10_000)

    tokeniser.create_encoding(
        data=get_text(dataset, train=True), 
        min_freq=5, 
        file_path=tokeniser_path
    )