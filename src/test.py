import torch 
import torch.nn as nn
from torch.nn import functional as F

from BPEtokeniser import BPETokeniser
from GPT2tokeniser import GPTtokenizer
from model import GPTModel 

def generate(model, idx, context_size, temperature, max_new_tokens):
    new_idx = []
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] 
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits / temperature, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        if idx_next == 0:
            return new_idx
        
        idx = torch.cat((idx, idx_next), dim=1)
        new_idx.append(idx_next)

    return new_idx

torch.manual_seed(411)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_dim = 512
context_size = 128
num_heads = 12
num_layers = 12

temperature = 0.7

tokeniser = GPTtokenizer()
model = GPTModel(tokeniser.vocab_size, embedding_dim, context_size, num_heads, num_layers, device, dropout=0.1)
checkpoint = torch.load('model_checkpoint.pth', map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
m = model.to(device)

conversation = ""
while True:
    conversation += input('YOU: ') + '\n'
    context = tokeniser.encode(conversation)[-context_size:]
    context = torch.tensor(context).unsqueeze(0)
    context.to(device)
    modeltext = tokeniser.decode(generate(model, context, context_size, temperature, max_new_tokens=1000)).split('\n')[0]
    conversation += modeltext + '\n'
    print('MicroGPT: ' + modeltext)
