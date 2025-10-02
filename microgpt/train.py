import torch
from torch.utils.data import DataLoader
import random
import numpy as np
import time
import argparse
import os

from microgpt.tokenizer import GPTtokenizer
from microgpt.config import Config

from microgpt.datasets.fineweb import FineWeb
from microgpt.datasets.hh_rlhf_chosen import HH_RLHF_Chosen
from microgpt.datasets.hh_rlhf import HH_RLHF

from microgpt.model.model import PretrainModel, FinetuneModel, RewardModel

def set_seed(seed=411):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def build_dataloader(dataset, batch_size, generator, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, generator=generator)

def save_checkpoint(model, optimizer, scheduler, scaler, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict()
    }
    torch.save(checkpoint, path)

def train(args):

    device = get_device()
    g = set_seed(args.seed)

    config = Config()
    tokenizer = GPTtokenizer()

    if args.mode == 'pretrain':
        dataset = FineWeb(tokenizer=tokenizer, context_size=config.context_size, device=device)
        val_dataset = None
        model = PretrainModel(config).to(device)

    elif args.mode == 'finetune':
        dataset = HH_RLHF_Chosen(tokenizer=tokenizer, context_size=config.context_size, device=device)
        val_dataset = HH_RLHF_Chosen(tokenizer=tokenizer, context_size=config.context_size, split='validation', device=device)
        model = FinetuneModel(config).to(device)

    elif args.mode == 'reward':
        dataset = HH_RLHF(tokenizer=tokenizer, context_size=config.context_size,save_path='hh_rlhf_tokens.bin', split='train', device=device)
        val_dataset = HH_RLHF(tokenizer=tokenizer, context_size=config.context_size, save_path='hh_rlhf_tokens_test.bin', split='test', device=device)
        model = RewardModel(config).to(device)

    total_steps = len(dataset) // args.batch_size
    
    if args.mode != 'pretrain' and args.model_load_path: # only pretrain will start without a checkpoint
        checkpoint = torch.load(args.model_load_path, weights_only=True)

        if args.mode == 'reward':
            del checkpoint["model_state_dict"]["transformer.lm_head.weight"]

        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=args.max_lr, total_steps=total_steps*args.epochs, pct_start=0.05, anneal_strategy='cos')
    scaler = torch.amp.GradScaler(device)

    batch_acc_steps = args.batch_size // args.batch_acc_size
    min_val_loss = float('inf')

    for epoch in range(args.epochs):

        dataloader = build_dataloader(dataset, args.batch_size, g)
        total_loss = 0.0
        counter = 0

        for current_batch, (xb, yb, mask) in enumerate(dataloader):

            start_time = time.time()
            optimizer.zero_grad(set_to_none=True)

            for i in range(batch_acc_steps):
                
                b_xb = xb[i*args.batch_acc_size:(i+1)*args.batch_acc_size].to(device, non_blocking=True)
                b_yb = yb[i*args.batch_acc_size:(i+1)*args.batch_acc_size].to(device, non_blocking=True)
                b_mask = mask[i*args.batch_acc_size:(i+1)*args.batch_acc_size].to(device, non_blocking=True) if args.mode == 'finetune' else None

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

            step_time = time.time() - start_time
            print(f'\repoch: {epoch+1}/{args.epochs} ' + \
                  f'| batch: {current_batch+1}/{total_steps} ' + \
                  f'| loss: {total_loss/(counter+1):.4f} ' + \
                  f'| lr: {scheduler.get_last_lr()[0]:.4e} ' + \
                  f'| step_time: {int(step_time*1000)}ms', end='') 

            counter += 1
            if args.mode == 'pretrain' and (current_batch + 1) % args.save_iter == 0:
                save_checkpoint(model, optimizer, scheduler, scaler, f'weights/base_model_checkpoint_{current_batch+1}.pth')
                total_loss = 0.0
                counter = 0

        if val_dataset:
            
            val_loss = 0.0
            val_dataloader = build_dataloader(val_dataset, args.batch_acc_size, g, shuffle=False)
            model.eval()
            with torch.no_grad():
                for xb, yb, mask in val_dataloader:
                    xb, yb, mask = xb.to(device), yb.to(device), mask.to(device)
                    _, loss = model(xb, yb, mask)
                    val_loss += loss.item()
            val_loss /= len(val_dataloader)
            print(f"\nepoch: {epoch+1}/{args.epochs} | validation loss: {val_loss:.4f}")
            model.train()

            if val_loss <= min_val_loss:
                min_val_loss = val_loss
                save_path = args.checkpoint_path.replace('.pth', f'_{epoch+1}.pth')
                save_checkpoint(model, optimizer, scheduler, scaler, save_path)

    torch.save(model.state_dict(), args.final_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['pretrain', 'finetune', 'reward'])
    parser.add_argument('--model_load_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=411)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--batch_acc_size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--max_lr', type=float, default=5e-4)
    parser.add_argument('--save_iter', type=int, default=5000)
    parser.add_argument('--checkpoint_path', type=str, default='weights/model.pth')
    parser.add_argument('--final_path', type=str, default='weights/model.pth')
    args = parser.parse_args()

    train(args)
