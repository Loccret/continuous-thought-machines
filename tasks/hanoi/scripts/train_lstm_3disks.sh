#!/bin/bash
RUN=1
HANOI_DISKS=3
LOG_DIR="logs/hanoi/run${RUN}/lstm_${HANOI_DISKS}disks"
SEED=$((RUN - 1))

python -m tasks.hanoi.train \
    --log_dir $LOG_DIR \
    --seed $SEED \
    --model_type lstm \
    --hanoi_disks $HANOI_DISKS \
    --d_model 1024 \
    --d_input 512 \
    --dropout 0.1 \
    --batch_size 128 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --num_epochs 100 \
    --grad_clip_quantile 0.90 \
    --warmup_epochs 5 \
    --scheduler_type cosine \
    --num_workers 4 \
    --pin_memory \
    --save_every 10 \
    --eval_every 1 \
    --log_every 100 \
    --save_gif_every 20 \
    --device auto