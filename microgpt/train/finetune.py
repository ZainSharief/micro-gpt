import torch
from torch.utils.data import DataLoader
import time

from microgpt.tokenizer import GPTtokenizer
from microgpt.config import Config
from microgpt.model.model import GPTModel
from microgpt.data.finetune_dataset import HH_RLHF_Chosen

def main():

    device = 'cpu'
    torch.manual_seed(411)  

    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.manual_seed(411)

    batch_size = 128
    batch_acc_size = 32
    batch_acc_steps = 4
    learning_rate = 3e-6
    max_lr = 5e-5
    epochs = 8

    # Load the dataset & tokeniser
    config = Config()
    tokeniser = GPTtokenizer()
    dataset = HH_RLHF_Chosen(tokenizer=tokeniser, context_size=config.context_size, device=device)
    val_dataset = HH_RLHF_Chosen(tokenizer=tokeniser, context_size=config.context_size, split='test', device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    total_steps = (len(dataset) // batch_size) * epochs

    model = GPTModel(config, use_lora=True)
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
    min_val_loss = float('inf')

    checkpoint = torch.load('weights/base_model_checkpoint_190000.pth', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    for epoch in range(epochs):
        total_loss = 0.0

        for current_batch, (xb, yb, loss_mask) in enumerate(dataloader):

            start_time = time.time()
            optimizer.zero_grad(set_to_none=True)

            for i in range(batch_acc_steps):
                
                b_xb = xb[i*batch_acc_size:(i+1)*batch_acc_size].to(device, non_blocking=True)
                b_yb = yb[i*batch_acc_size:(i+1)*batch_acc_size].to(device, non_blocking=True)
                b_mask = loss_mask[i*batch_acc_size:(i+1)*batch_acc_size].to(device, non_blocking=True)

                with torch.autocast(device_type=device, dtype=torch.float16):
                    _, loss = model(b_xb, b_yb, b_mask)
                    loss = loss / batch_acc_steps

                scaler.scale(loss).backward()
                total_loss += loss.item()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_time = time.time() - start_time
        
            print(f"\repoch: {epoch+1}/{epochs} | batch: {current_batch+1}/{len(dataloader)} | loss: {(total_loss/(current_batch+1)):.4f} | lr: {scheduler.get_last_lr()[0]:.4e} | step_time: {int(total_time*1000)}ms", end='') 
            
        total_loss = 0.0

        val_loss = 0.0
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        model.eval()
        with torch.no_grad():
            for xb, yb, mask in val_dataloader:
                xb, yb, mask = xb.to(device), yb.to(device), mask.to(device)
                _, loss = model(xb, yb, mask)
                val_loss += loss.item()
        val_loss /= len(val_dataloader)
        print(f"\nepoch: {epoch+1}/{epochs} | validation loss: {val_loss:.4f}")
        model.train()

        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }
            torch.save(checkpoint, f'model_checkpoint.pth')
        total_loss = 0.0
           
    torch.save(model.state_dict(), config.fine_tuned_model_path)

if __name__ == '__main__':
    main()