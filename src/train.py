import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import time

from GPT2tokeniser import GPTtokenizer
from dataset import FineWeb
from model import GPTModel

device = 'cpu'
torch.manual_seed(411)

if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.manual_seed(411)

@dataclass
class config:
    embedding_dim: int = 1024
    context_size: int = 256
    num_heads: int = 16
    num_layers: int = 12
    max_norm: float = 1.0

def main():
    
    batch_size = 8                          # Number of samples in each batch (16 to prevent CUDA memory errors)
    grad_acc_size = 4
    learning_rate = 2e-4
    max_lr = 4e-4
    inference_iter = 5_000                  # Number of iterations before inference
    save_iter = 50_000                      # Number of iterations before saving the model

    # Load the dataset & tokeniser
    tokeniser = GPTtokenizer()
    dataset = FineWeb(tokeniser=tokeniser, context_size=config.context_size, batch_size=batch_size, device=device) 
    total_steps = len(dataset) - (len(dataset) % grad_acc_size)

    model = GPTModel(tokeniser.vocab_size, config.embedding_dim, config.context_size, config.num_heads, config.num_layers, device=device, dropout=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_lr,
        total_steps=total_steps//grad_acc_size,
        pct_start=0.3, 
        anneal_strategy='cos'
    )
    scaler = torch.amp.GradScaler(device) 
    model = model.to(device)
    model = torch.compile(model)

    for current_batch in range(0, total_steps, grad_acc_size):

        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)

        for current_step in range(current_batch, current_batch+grad_acc_size):

            # Collect the sample of data for that batch
            xb, yb = dataset.__getbatch__(current_step)

            with torch.amp.autocast(device, dtype=torch.float16):
                _, loss = model(xb, yb)
            scaler.scale(loss / grad_acc_size).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)

        scaler.step(optimizer)
        scheduler.step()
        scaler.update()

        total_time = time.time() - start_time
    
        print(f"\rbatch: {(current_batch//grad_acc_size)+1}/{total_steps//grad_acc_size} | loss: {loss:.4f} | lr: {scheduler.get_last_lr()[0]:.4e} | step_time: {int(total_time*1000)}ms", end='') 
        
        if (current_batch + 1) % inference_iter == 0:
            print('\n' + model.generate(tokeniser, '', temperature=0.7, k=20, max_new_tokens=100, device=device))

        if (current_batch + 1) % save_iter == 0:

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),                
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }
            torch.save(checkpoint, f'model_checkpoint{current_step+1}.pth')

    torch.save(model.state_dict(), 'model_final.pth')

if __name__ == '__main__':
    main()