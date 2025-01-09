import torch
import torch.nn as nn
from torch.nn import functional as F

from datasets import load_dataset

from GPT2tokeniser import GPTtokenizer
from model import GPTModel
from preprocess import get_batch

def calculate_loss(xb, yb):
    B, T, C = xb.shape
    xb = xb.view(B*T, C)
    yb = yb.view(B*T)
    loss = F.cross_entropy(xb, yb)
    return loss

def calculate_val_loss(val_size):
    with torch.no_grad():
        val_loss = 0
        for _ in range(val_size):
            xb, yb = get_batch(dataset, tokeniser, batch_size, context_size, device, train=False)
            val_data = model(xb)
            val_loss += calculate_loss(val_data, yb)

        return (val_loss / val_size)

torch.manual_seed(411)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32
embedding_dim = 512
context_size = 128
num_heads = 8
num_layers = 8
learning_rate = 1e-4
max_iters = 11118
val_iter = 1000

# Load the dataset
dataset = load_dataset("daily_dialog", trust_remote_code=True)
tokeniser = GPTtokenizer()

model = GPTModel(tokeniser.vocab_size, embedding_dim, context_size, num_heads, num_layers, device, dropout=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

min_val_loss = float('inf')
m = model.to(device)

total_loss = 0
for iter in range(max_iters):

    # sample a batch of data
    xb, yb = get_batch(dataset, tokeniser, batch_size, context_size, device, iter=iter)
    optimizer.zero_grad()

    # evaluate the loss
    out = model(xb)

    loss = calculate_loss(out, yb)
    total_loss += loss
    loss.backward()

    if (iter + 1) % val_iter == 0:
        val_loss = calculate_val_loss(val_size=100)

        if  min_val_loss > val_loss:
            min_val_loss = val_loss

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, 'model_checkpoint.pth')

        scheduler.step()

        print(f"\rbatch={iter+1}/{max_iters} loss={total_loss/(iter+1):.4f} val_loss={val_loss:.4f}")

    print(f"\rbatch={iter+1}/{max_iters} loss={total_loss/(iter+1):.4f}", end='')

    optimizer.step()

torch.save(model.state_dict(), 'model_weights.pth')