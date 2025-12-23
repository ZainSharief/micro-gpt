#!/bin/bash
python -u microgpt/train.py \
    --mode finetune \
    --dropout 0.1 \
    --model_load_path weights/base_model_checkpoint_190000.pth \
    --seed 411 \
    --epochs 8 \
    --batch_size 128 \
    --batch_acc_size 32 \
    --weight_decay 0.0 \
    --lr 5e-5 \
    --max_lr 2e-4 \
    --checkpoint_path weights/finetune_checkpoint.pth \
    --final_path weights/fine_tuned_model.pth