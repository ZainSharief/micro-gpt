#!/bin/bash
python -u microgpt/train.py \
    --mode pretrain \
    --dropout 0.0 \
    --seed 411 \
    --epochs 1 \
    --batch_size 128 \
    --batch_acc_size 32 \
    --weight_decay 0.01 \
    --lr 5e-5 \
    --max_lr 3e-4 \
    --save_iter 5000 \
    --checkpoint_path weights/pretrain_checkpoint.pth \
    --final_path weights/base_model.pth