#!/bin/bash
RUN=1
ITERATIONS=75
MEMORY_LENGTH=25
RIVER_PAIRS=4
LOG_DIR="logs/river_crossing/run${RUN}/ctm_${ITERATIONS}_${MEMORY_LENGTH}_${RIVER_PAIRS}pairs"
SEED=$((RUN - 1))

conda run --name ctm python -m tasks.river_crossing.train \
    --log_dir $LOG_DIR \
    --seed $SEED \
    --model_type ctm \
    --river_pairs $RIVER_PAIRS \
    --d_model 2048 \
    --d_input 512 \
    --iterations $ITERATIONS \
    --memory_length $MEMORY_LENGTH \
    --heads 16 \
    --n_synch_out 64 \
    --n_synch_action 32 \
    --synapse_depth 8 \
    --deep_memory \
    --memory_hidden_dims 32 \
    --backbone_type parity_backbone \
    --positional_embedding_type none \
    --dropout 0.1 \
    --no-do_normalisation \
    --neuron_select_type first-last \
    --n_random_pairing_self 0 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --num_epochs 120 \
    --grad_clip_quantile -1 \
    --warmup_epochs 12 \
    --scheduler_type cosine \
    --num_workers 4 \
    --pin_memory \
    --save_every 10 \
    --eval_every 1 \
    --log_every 100 \
    --save_gif_every 20 \
    --device auto