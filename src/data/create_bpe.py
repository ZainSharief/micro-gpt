from dataset import Dataset
from BPEtokeniser import BPETokeniser

tokeniser = BPETokeniser(num_merges=20_000)
dataset = Dataset(tokeniser=tokeniser, context_size=None, batch_size=None, device=None, train=True)

tokeniser_path = 'byte-pair-encoding20000.pkl'

tokeniser.create_encoding(
    data=dataset.get_text(), 
    min_freq=5, 
    file_path=tokeniser_path
)

print('\rTokeniser has been created.')