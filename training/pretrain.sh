#!/bin/bash
python -u microgpt/train.py \
    --mode pretrain \
    --seed 411 \
    --epochs 1 \
    --batch_size 128 \
    --batch_acc_size 32 \
    --weight_decay 0.01 \
    --lr 3e-5 \
    --max_lr 5e-4 \
    --save_iter 5000 \
    --pretrain_path weights/base_model.pth