#!/bin/bash
python -u microgpt/train.py \
    --mode reward \
    --model_load_path weights\hh_rlhf_chosen_finetune.pth \
    --seed 411 \
    --epochs 3 \
    --batch_size 64 \
    --batch_acc_size 32 \
    --weight_decay 0.001 \
    --lr 5e-5 \
    --max_lr 4e-4 \
    --checkpoint_path weights/reward_model.pth \
    --final_path weights/reward_model.pth