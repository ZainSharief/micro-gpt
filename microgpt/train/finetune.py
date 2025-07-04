import torch
from torch.utils.data import DataLoader
import time

from microgpt.tokenizer import GPTtokenizer
from microgpt.config import Config
from microgpt.model.model import GPTModel
from microgpt.data.finetune_dataset import OpenAssistantDataset

def main():

    device = 'cpu'
    torch.manual_seed(411)  

    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.manual_seed(411)

    batch_size = 32                        # Number of samples in each batch 
    learning_rate = 2e-4
    max_lr = 6e-4
    epochs = 100

    # Load the dataset & tokeniser
    config = Config()
    tokeniser = GPTtokenizer()
    dataset = OpenAssistantDataset(tokeniser=tokeniser, context_size=config.context_size, device=device) 
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model = GPTModel(config, use_lora=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_lr,
        steps_per_epoch=len(dataloader),
        epochs=epochs,
        pct_start=0.1, 
        anneal_strategy='cos'
    )
    scaler = torch.amp.GradScaler(device)
    model = model.to(device)
    lowest_loss = float('inf')

    for epoch in range(epochs):

        total_loss = 0.0

        for current_batch, (xb, yb, loss_mask) in enumerate(dataloader):

            start_time = time.time()
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device, dtype=torch.float16):
                _, loss = model(xb, yb, loss_mask=loss_mask)
                total_loss += loss.item()
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_time = time.time() - start_time
        
            print(f"\repoch: {epoch+1}/{epochs} | batch: {current_batch+1}/{len(dataloader)} | loss: {(total_loss/(current_batch+1)):.4f} | lr: {scheduler.get_last_lr()[0]:.4e} | step_time: {int(total_time*1000)}ms", end='') 
            
        print()
        if (total_loss/(current_batch+1)) < lowest_loss:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }
            torch.save(checkpoint, f'model_checkpoint.pth')
                    
    torch.save(model.state_dict(), config.fine_tuned_model_path)

if __name__ == '__main__':
    main()