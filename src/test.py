import torch 
import torch.nn as nn
from torch.nn import functional as F

from data.GPT2tokeniser import GPTtokenizer
from model import GPTModel 

torch.manual_seed(411)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_dim = 768
context_size = 256
num_heads = 12
num_layers = 12

temperature = 0.8
k = 15

tokeniser = GPTtokenizer()
model = GPTModel(tokeniser.vocab_size, embedding_dim, context_size, num_heads, num_layers, device, dropout=0.2)
m = model.to(device)

checkpoint = torch.load('model_checkpoint.pth', map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

conversation = ""
while True:

    user_text = input('YOU: ')

    if user_text != '':

        conversation += user_text 

        model_text = model.generate(
            tokeniser=tokeniser, 
            text=conversation, 
            temperature=temperature, 
            k=k,
            max_new_tokens=100,
        )

        conversation += '<|endoftext|>' + model_text + '<|endoftext|>'

        print('MicroGPT: ' + model_text)