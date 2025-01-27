import torch
import torch.nn as nn
from torch.nn import functional as F

from data.GPT2tokeniser import GPTtokenizer
from data.dataset import Dataset
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
        val_dataset = Dataset(tokeniser=tokeniser, context_size=context_size, batch_size=batch_size, device=device, train=False) 
        for idx in range(val_size // batch_size):
            xb, yb = val_dataset.__getitem__(idx)
            val_data = model(xb)
            val_loss += calculate_loss(val_data, yb)

        return (val_loss / (val_size // batch_size))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(411)
if device == 'cuda':
    torch.cuda.manual_seed(411)

embedding_dim = 768
context_size = 256
num_heads = 12
num_layers = 12

batch_size = 32
learning_rate = 3e-4
epochs = 5
max_iters = 11118       # Number of training samples
val_iter = 1000         # Number of iterations before each validation
val_size = 100          # Number of samples to validate

# Load the dataset & tokeniser
tokeniser = GPTtokenizer()
dataset = Dataset(tokeniser=tokeniser, context_size=context_size, batch_size=batch_size, device=device) 

model = GPTModel(tokeniser.vocab_size, embedding_dim, context_size, num_heads, num_layers, device, dropout=0.2)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=learning_rate,
    total_steps=max_iters*epochs,
    pct_start=0.3, 
    anneal_strategy='cos'
)
m = model.to(device)

min_val_loss = float('inf')

for epoch in range(epochs):

    total_loss = 0
    epoch_iter = 0

    for current_iter in range(max_iters // batch_size):

        # Collect the sample of data for that batch
        xb, yb = dataset.__getbatch__(current_iter)

        # Completes forward & backward pass
        optimizer.zero_grad()

        out = model(xb)
        loss = calculate_loss(out, yb)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Uses total loss to get a smoother loss curve
        epoch_iter += 1 
        total_loss += loss
        print(f"\rbatch={current_iter+1}/{max_iters} loss={total_loss/(epoch_iter):.4f}", end='')

        if (current_iter + 1) % val_iter == 0:      

            val_loss = calculate_val_loss(val_size=val_size)

            # Only saves the model if it performs better than the previous best
            if  min_val_loss > val_loss:
                min_val_loss = val_loss

                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }
                torch.save(checkpoint, 'model_checkpoint.pth')

            print(f"\rbatch={current_iter+1}/{max_iters} loss={total_loss/(epoch_iter):.4f} val_loss={val_loss:.4f}")

            # Resets the total loss and epoch_iter after every validation 
            # Otherwise total loss is skewed by earlier batches
            total_loss = 0
            epoch_iter = 0
            
    print('')

torch.save(model.state_dict(), 'model_weights.pth')