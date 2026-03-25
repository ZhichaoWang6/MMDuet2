#!/bin/bash
# Step 2: Train the Kangaroo adapter
# Only trains a lightweight adapter (1 transformer layer) with KL-divergence loss.
# Base model weights are frozen.

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # or your fine-tuned checkpoint
DATA_DIR=./training_data/
OUTPUT_DIR=./adapter_checkpoints/
EXIT_LAYER=2

cd "$(dirname "$0")/.."

accelerate launch train_adapter.py \
    --basepath $MODEL_PATH \
    --datadir $DATA_DIR \
    --outdir $OUTPUT_DIR \
    --exit_layer $EXIT_LAYER \
    --num_adapter_layers 1 \
    --lr 3e-5 \
    --bs 4 \
    --gradient_accumulation_steps 8 \
    --num_epochs 20 \
    --num_warmup_steps 2000 \
    --max_len 2048 \
    --grad_clip 0.5 \
    --save_freq 2
