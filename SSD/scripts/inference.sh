#!/bin/bash
# Step 3: Run inference with Kangaroo speculative decoding

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # or your fine-tuned checkpoint
ADAPTER_PATH=./adapter_checkpoints/adapter_epoch_0  # trained adapter
EXIT_LAYER=2
SPECULATIVE_STEPS=6
THRESHOLD=0.6

cd "$(dirname "$0")/.."

python inference_example.py \
    --model_path $MODEL_PATH \
    --adapter_path $ADAPTER_PATH \
    --exit_layer $EXIT_LAYER \
    --speculative_steps $SPECULATIVE_STEPS \
    --threshold $THRESHOLD \
    --prompt "Describe what you see in this image."
