import torch 
import torch.nn as nn
from torch.nn import functional as F

from data.BPEtokeniser import BPETokeniser
from model import GPTModel 

torch.manual_seed(411)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_dim = 768
context_size = 128
num_heads = 12
num_layers = 12

temperature = 0.7
k=15

tokeniser = BPETokeniser(10_000)
tokeniser.load(file_path='byte-pair-encoding10000.pkl')
model = GPTModel(tokeniser.vocab_size, embedding_dim, context_size, num_heads, num_layers, device, dropout=0.2)
m = model.to(device)

checkpoint = torch.load('model_checkpoint.pth', map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

conversation = ""
while True:
    
    user_text = input('YOU: ')

    if user_text != '':

        conversation += user_text + '\n'

        model_text = model.generate(
            tokeniser=tokeniser, 
            text=conversation, 
            temperature=temperature, 
            k=k,
            max_new_tokens=100,
        )

        print('MicroGPT: ' + model_text)