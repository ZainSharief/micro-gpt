#!/bin/bash
python -u microgpt/train.py \
    --mode reward \
    --model_load_path weights\hh_rlhf_chosen_finetune.pth \
    --seed 411 \
    --epochs 6 \
    --batch_size 128 \
    --batch_acc_size 32 \
    --weight_decay 0.01 \
    --lr 1e-3 \
    --max_lr 5e-4 \
    --checkpoint_path weights/reward_model.pth \
    --final_path weights/reward_model.pth