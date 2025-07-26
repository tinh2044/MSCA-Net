#!/bin/bash

BATCH_SIZE=1
EPOCHS=100
FINETUNE=""
DEVICE="cpu"
SEED=0
RESUME=""
RESUME="./outputs/Phoenix-2014-T/best_checkpoint.pth"
START_EPOCH=0
EVAL_FLAG=True
TEST_ON_LAST_EPOCH="False"
NUM_WORKERS=0
CFG_PATH="configs/phoenix-2014t.yaml"

# Attention map specific settings
GENERATE_ATTENTION_MAPS=True
ATTENTION_OUTPUT_DIR="./attention_maps/Phoenix-2014-T"
MAX_SAMPLES=50  # Limit number of samples to process for attention maps

python -m main \
   --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --finetune "$FINETUNE" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --resume "$RESUME" \
    --start_epoch "$START_EPOCH" \
    --eval \
    --test_on_last_epoch "$TEST_ON_LAST_EPOCH" \
    --num_workers "$NUM_WORKERS" \
    --cfg_path "$CFG_PATH" \
    --generate_attention_maps \
    --attention_output_dir "$ATTENTION_OUTPUT_DIR" \
    --max_samples "$MAX_SAMPLES" 