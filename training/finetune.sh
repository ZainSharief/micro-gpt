#!/bin/bash
python -u microgpt/train.py \
    --mode finetune \
    --dropout 0.05 \
    --model_load_path model/pretrain_model.pth \
    --seed 42 \
    --epochs 5 \
    --batch_size 16 \
    --batch_acc_size 8 \
    --weight_decay 0.01 \
    --lr 1e-5 \
    --embedding_lr 1e-5 \
    --max_lr 5e-5 \
    --embedding_max_lr 5e-5 \
    --validation_iter 100 \
    --checkpoint_path weights/finetune_checkpoint.pth \
    --final_path weights/fine_tuned_model.pth
