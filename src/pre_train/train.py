import torch
from torch.utils.data import DataLoader
import time

from src.tokenizer import GPTtokenizer
from src.config import config
from dataset import FineWeb
from model import GPTModel

def main():

    device = 'cpu'
    torch.manual_seed(411)  

    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.manual_seed(411)

    batch_size = 32                        # Number of samples in each batch 
    learning_rate = 2e-4
    max_lr = 6e-4
    inference_iter = 5_000                 # Number of iterations before inference
    save_iter = 10_000                     # Number of iterations before saving the model

    # Load the dataset & tokeniser
    tokeniser = GPTtokenizer()
    dataset = FineWeb(tokenizer=tokeniser, context_size=config.context_size, device=device) 
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model = GPTModel(tokeniser.vocab_size, config.embedding_dim, config.context_size, config.num_heads, config.num_layers, device=device, dropout=config.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_lr,
        total_steps=1_000_000,
        pct_start=0.1, 
        anneal_strategy='cos'
    )
    scaler = torch.amp.GradScaler(device)
    model = model.to(device)
    
    for current_batch, (xb, yb) in enumerate(dataloader):

        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device, dtype=torch.float16):
            _, loss = model(xb, yb)
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_time = time.time() - start_time
    
        print(f"\rbatch: {current_batch+1}/1,000,000 | loss: {loss:.4f} | lr: {scheduler.get_last_lr()[0]:.4e} | step_time: {int(total_time*1000)}ms", end='') 
        
        if (current_batch + 1) % inference_iter == 0:
            print('\n' + model.generate(tokeniser, 'When I go to the shops, I usually buy', temperature=config.temperature, k=config.k, max_new_tokens=100, device=device))

        if (current_batch + 1) % save_iter == 0:

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),                
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }
            torch.save(checkpoint, f'model_checkpoint{current_batch+1}.pth')
                
    torch.save(model.state_dict(), config.base_model_path)

if __name__ == '__main__':
    main()