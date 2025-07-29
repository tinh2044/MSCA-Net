#!/bin/bash

# Script to visualize attention maps from MSCA-Net model
# Usage: bash scripts/visualize_attention.sh

# Configuration
CONFIG_PATH="configs/phoenix-2014t.yaml"
CHECKPOINT_PATH="./outputs/Phoenix-2014-T/checkpoint_344.pth"  # Update this path
OUTPUT_DIR="./attention_maps"
SPLIT="test"
MAX_SAMPLES=300  # Limit number of samples to process
BATCH_SIZE=1    
DEVICE="cpu"    

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    echo "Please update CONFIG_PATH in this script"
    exit 1
fi

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found at $CHECKPOINT_PATH"
    echo "Please update CHECKPOINT_PATH in this script with the path to your trained model"
    exit 1
fi

echo "Starting attention visualization..."
echo "Config: $CONFIG_PATH"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Split: $SPLIT"
echo "Max samples: $MAX_SAMPLES"
echo "Batch size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo ""

# Run visualization
python visualize_attention.py \
    --cfg_path "$CONFIG_PATH" \
    --checkpoint "$CHECKPOINT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --split "$SPLIT" \
    --max_samples "$MAX_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --seed 42

echo ""
echo "Visualization completed!"
echo "Check the output directory: $OUTPUT_DIR"
