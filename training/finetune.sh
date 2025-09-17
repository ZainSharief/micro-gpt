#!/bin/bash
python -u microgpt/train.py \
    --mode finetune \
    --checkpoint_path weights/base_model_checkpoint_190000.pth \
    --seed 411 \
    --epochs 8 \
    --batch_size 128 \
    --batch_acc_size 32 \
    --weight_decay 0.0 \
    --lr 3e-6 \
    --max_lr 5e-5 \
    --finetune_path weights/fine_tuned_model.pth

