#!/bin/bash
# Single-node multi-GPU training script

# 设置可见GPU (可选，如果想指定特定GPU)
# export CUDA_VISIBLE_DEVICES=0,1  # 使用GPU 0和1

# 使用accelerate启动训练
accelerate launch \
    --config_file training/configs/accelerate_single_node.yaml \
    train.py \
    --config configs/base_config.yaml

# 说明：
# - accelerate会自动使用配置文件中指定的GPU数量
# - 每个GPU会加载batch_size=4的数据
# - 实际总batch_size = 4 × 2 GPUs = 8
# - Steps per epoch会减少一半，但总数据量不变
