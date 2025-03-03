import torch
import torch.nn as nn
from torch.nn import functional as F
import time

from config import config
from tokenizer import GPTtokenizer
from dataset import FineWeb
from model import GPTModel

device = 'cpu'
torch.manual_seed(411)

if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.manual_seed(411)

def main():
    
    batch_size = 8                          # Number of samples in each batch (16 to prevent CUDA memory errors)
    grad_acc_size = 8
    learning_rate = 2e-4
    max_lr = 4e-4
    inference_iter = 1_000                  # Number of iterations before inference
    save_iter = 5_000                       # Number of iterations before saving the model

    # Load the dataset & tokeniser
    tokeniser = GPTtokenizer()
    dataset = FineWeb(tokeniser=tokeniser, context_size=config.context_size, batch_size=batch_size, size=172_000*grad_acc_size, device=device) 
    total_steps = len(dataset) - (len(dataset) % grad_acc_size)

    model = GPTModel(tokeniser.vocab_size, config.embedding_dim, config.context_size, config.num_heads, config.num_layers, device=device, dropout=config.dropout)
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

    for current_batch in range(0, total_steps, grad_acc_size):

        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)

        for current_step in range(current_batch, current_batch+grad_acc_size):

            xb, yb = dataset.__nextbatch__()

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
        
        if ((current_batch//grad_acc_size) + 1) % inference_iter == 0:
            print('\n' + model.generate(tokeniser, 'The best way to greet someone is to say', temperature=config.temperature, k=config.k, max_new_tokens=100, device=device))

        if ((current_batch//grad_acc_size) + 1) % save_iter == 0:

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