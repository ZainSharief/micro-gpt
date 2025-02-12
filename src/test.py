import torch 
import torch.nn as nn
from torch.nn import functional as F

from GPT2tokeniser import GPTtokenizer
from model import GPTModel
from train import config

device = 'cpu'
torch.manual_seed(411)
if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.manual_seed(411)

temperature = 0.7
k = 20

tokeniser = GPTtokenizer()
model = GPTModel(tokeniser.vocab_size, config.embedding_dim, config.context_size, config.num_heads, config.num_layers, device, dropout=0.2)
m = model.to(device)

checkpoint = torch.load('model_checkpoint200000.pth', map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

conversation = ''
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

        conversation += model_text

        print('MicroGPT: ' + model_text)