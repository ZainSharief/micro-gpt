#!/bin/bash
python -u microgpt/train.py \
    --mode finetune \
    --dropout 0.05 \
    --model_load_path model/pretrain_model.pth \
    --seed 42 \
    --epochs 3 \
    --batch_size 128 \
    --batch_acc_size 8 \
    --weight_decay 0.1 \
    --lr 1e-4 \
    --max_lr 2e-4 \
    --validaton_iter 100 \
    --checkpoint_path weights/finetune_checkpoint.pth \
    --final_path weights/fine_tuned_model.pth