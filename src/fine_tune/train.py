import torch
from torch.utils.data import DataLoader
import time

from src.tokenizer import GPTtokenizer
from src.config import config
from dataset import OpenAssistant
from model import GPTModel

def main():

    device = 'cpu'
    torch.manual_seed(411)  

    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.manual_seed(411)

    batch_size = 32                         # Number of samples in each batch (16 to prevent CUDA memory errors)
    learning_rate = 5e-6
    max_lr = 1e-5
    epochs = 10

    # Load the dataset & tokeniser
    tokeniser = GPTtokenizer()
    dataset = OpenAssistant(tokeniser=tokeniser, context_size=config.context_size, batch_size=batch_size, device=device) 
    total_steps = (len(dataset) // batch_size) if len(dataset) % batch_size == 0 else (len(dataset) // batch_size) + 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GPTModel(tokeniser.vocab_size, config.embedding_dim, config.context_size, config.num_heads, config.num_layers, device=device, dropout=config.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_lr,
        total_steps=epochs*total_steps,
        pct_start=0.1, 
        anneal_strategy='cos'
    )
    scaler = torch.amp.GradScaler(device)
    model = model.to(device)

    model.load_state_dict(torch.load(config.base_model_path, map_location=device, weights_only=True))

    for epoch in range(epochs):
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
    
            print(f"\repoch: {epoch+1}/{epochs} | batch: {current_batch+1}/{total_steps} | loss: {loss:.4f} | lr: {scheduler.get_last_lr()[0]:.4e} | step_time: {int(total_time*1000)}ms", end='') 
        
        print('\n' + model.generate(tokeniser, 'What is the best way to greet someone?', temperature=config.temperature, k=config.k, max_new_tokens=100, device=device))
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),                
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict()
        }
        torch.save(checkpoint, f'model_checkpoint{current_batch+1}.pth')
                
    torch.save(model.state_dict(), config.fine_tuned_model_path)

if __name__ == '__main__':
    main()