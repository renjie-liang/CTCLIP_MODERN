#!/bin/bash
# Single-node multi-GPU training script

# Set visible GPUs (optional, if you want to specify particular GPUs)
# export CUDA_VISIBLE_DEVICES=0,1  # Use GPU 0 and 1

# Launch training with accelerate
accelerate launch \
    --config_file run_setup/configs/accelerate_single_node.yaml \
    train.py \
    --config configs/base_config.yaml

# Notes:
# - Accelerate will automatically use the number of GPUs specified in the config file
# - Each GPU will load batch_size=4 data
# - Effective total batch_size = 4 × 2 GPUs = 8
# - Steps per epoch will be halved, but total data volume remains the same
