#!/bin/bash
python -u microgpt/train.py \
    --mode reward \
    --checkpoint_path weights/fine_tuned_checkpoint_8.pth \
    --seed 411 \
    --epochs 3 \
    --batch_size 128 \
    --batch_acc_size 32 \
    --weight_decay 0.01 \
    --lr 1e-5 \
    --max_lr 5e-5 \
    --final_path weights/reward_model.pth