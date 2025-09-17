import torch
from torch.utils.data import DataLoader
import time

from microgpt.tokenizer import GPTtokenizer
from microgpt.config import Config
from microgpt.data.dataset import FineWeb
from microgpt.model import GPTModel

def main():

    device = 'cpu'
    torch.manual_seed(411)  

    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.manual_seed(411)

    batch_size = 128 
    batch_acc_size = 32       
    batch_acc_steps = 4  
    learning_rate = 3e-5
    max_lr = 5e-4
    save_iter = 5_000                     
    
    # Load the dataset & tokeniser
    config = Config()
    tokeniser = GPTtokenizer()
    dataset = FineWeb(tokenizer=tokeniser, context_size=config.context_size, device=device) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    total_steps = len(dataset) // batch_size

    model = GPTModel(config, use_lora=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=0.1, 
        anneal_strategy='cos'
    )
    scaler = torch.amp.GradScaler(device)
    model = model.to(device)
    
    for current_batch, (xb, yb) in enumerate(dataloader):

        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for i in range(batch_acc_steps):
                
                b_xb = xb[i*batch_acc_size:(i+1)*batch_acc_size].to(device, non_blocking=True)
                b_yb = yb[i*batch_acc_size:(i+1)*batch_acc_size].to(device, non_blocking=True)

                with torch.autocast(device_type=device, dtype=torch.float16):
                    _, loss = model(b_xb, b_yb)
                    loss = loss / batch_acc_steps

                scaler.scale(loss).backward()
                total_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_time = time.time() - start_time
    
        print(f"\rbatch: {current_batch+1}/{total_steps} | loss: {(total_loss/batch_acc_steps):.4f} | lr: {scheduler.get_last_lr()[0]:.4e} | step_time: {int(total_time*1000)}ms", end='') 

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