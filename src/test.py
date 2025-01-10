import torch 
import torch.nn as nn
from torch.nn import functional as F

from data.BPEtokeniser import BPETokeniser
from model import GPTModel 

@torch.no_grad()
def generate(model, idx, context_size, temperature, max_new_tokens):
    new_idx = []
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] 
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        print(probs)
        idx_next = torch.multinomial(probs, num_samples=1)

        if idx_next == 0:
            return new_idx
        
        idx = torch.cat((idx, idx_next), dim=1)
        new_idx.append(idx_next)

    return new_idx

torch.manual_seed(411)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_dim = 768
context_size = 256
num_heads = 12
num_layers = 12

temperature = 0.7

tokeniser = BPETokeniser(10_000)
tokeniser.load(file_path='byte-pair-encoding10000.pkl')
model = GPTModel(tokeniser.vocab_size, embedding_dim, context_size, num_heads, num_layers, device, dropout=0.2)
m = model.to(device)

checkpoint = torch.load('model_checkpoint.pth', map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

conversation = ""
while True:
    conversation += input('YOU: ') + '\n'
    context = tokeniser.encode(conversation)[-context_size:] 
    context = context + [0]*(context_size - len(context))
    context = torch.tensor(context).unsqueeze(0)
    context.to(device)
    modeltext = tokeniser.decode(generate(model, context, context_size, temperature, max_new_tokens=100)).split('\n')[0]
    conversation += modeltext + '\n'
    print('MicroGPT: ' + modeltext)
