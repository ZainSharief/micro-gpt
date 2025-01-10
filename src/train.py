import torch
import torch.nn as nn
from torch.nn import functional as F

from datasets import load_dataset

from data.BPEtokeniser import BPETokeniser
from data.preprocess import get_batch
from model import GPTModel

def calculate_loss(xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    B, T, C = xb.shape
    xb = xb.view(B*T, C)
    yb = yb.view(B*T)
    loss = F.cross_entropy(xb, yb, ignore_index=0)
    return loss

def calculate_val_loss(val_size: int) -> torch.Tensor:
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
embedding_dim = 768
context_size = 128   
num_heads = 12
num_layers = 12
learning_rate = 1e-4
max_iters = 11118       # Number of training samples
val_iter = 1000         # Number of iterations before each validation
val_size = 100          # Number of samples to validate 

# Load the dataset & tokeniser
dataset = load_dataset("daily_dialog", trust_remote_code=True)
tokeniser = BPETokeniser(num_merges=10_000)
tokeniser.load(file_path='byte-pair-encoding10000.pkl')

model = GPTModel(tokeniser.vocab_size, embedding_dim, context_size, num_heads, num_layers, device, dropout=0.2)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
scaler = torch.amp.GradScaler(device=device)
m = model.to(device)

min_val_loss = float('inf')
total_loss = 0
epoch_iter = 0

for iter in range(max_iters):

    # Collect the sample of data for that batch
    xb, yb = get_batch(dataset, tokeniser, batch_size, context_size, device, iter=iter)

    # Completes forward & backward pass
    optimizer.zero_grad()
    with torch.amp.autocast(device):
        out = model(xb)
        loss = calculate_loss(out, yb)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # Uses total loss to get a smoother loss curve
    epoch_iter += 1 
    total_loss += loss
    print(f"\rbatch={iter+1}/{max_iters} loss={total_loss/(epoch_iter):.4f}", end='')

    if (iter + 1) % val_iter == 0:      

        val_loss = calculate_val_loss(val_size=val_size)

        # Only saves the model if it performs better than the previous best
        if  min_val_loss > val_loss:
            min_val_loss = val_loss

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }
            torch.save(checkpoint, 'model_checkpoint.pth')

        print(f"\rbatch={iter+1}/{max_iters} loss={total_loss/(epoch_iter):.4f} val_loss={val_loss:.4f}")

        # Resets the total loss and epoch_iter after every validation 
        # Otherwise total loss is skewed by earlier batches
        total_loss = 0
        epoch_iter = 0

        scheduler.step((iter + 1) // val_iter)

torch.save(model.state_dict(), 'model_weights.pth')