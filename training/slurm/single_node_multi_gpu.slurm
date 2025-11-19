#!/bin/bash

#SBATCH --job-name=ctclip_train_multi_gpu
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:4              # 使用4个GPU (可以改成2/8等)
#SBATCH --nodes=1                 # 单节点
#SBATCH --cpus-per-task=32        # 建议增加CPU以匹配GPU数量
#SBATCH --mem=400gb               # 建议增加内存以匹配GPU数量
#SBATCH --time=72:00:00
#SBATCH --account=xujie
#SBATCH --qos=xujie
#SBATCH --output=out_slurm/train_multi_gpu_%j.out
#SBATCH --error=out_slurm/train_multi_gpu_%j.err

# Create output directories
mkdir -p out_slurm
mkdir -p logs
mkdir -p saves

# Setup Micromamba environment
export MAMBA_EXE='/home/liang.renjie/micromamba'
export MAMBA_ROOT_PREFIX='/blue/xujie/liang.renjie/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate b200

# Change to project directory
cd /orange/xujie/liang.renjie/3DCT/CTCLIP_MODERN

# Print environment information (for debugging)
echo "========================================"
echo "Job Information"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory: 400GB"
echo "GPUs: 4 (CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES)"
echo "========================================"
echo "Environment"
echo "========================================"
echo "Working directory: $(pwd)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "========================================"
echo "Training Configuration"
echo "========================================"
echo "Config file: configs/base_config.yaml"
echo "Accelerate config: training/configs/accelerate_single_node.yaml"
echo "Data format: WebDataset (float16)"
echo "Mixed precision: Enabled (fp16)"
echo "Batch size per GPU: 8"
echo "Number of GPUs: 4"
echo "Effective batch size: 32 (8 x 4)"
echo "Num workers: 24"
echo "Max steps: 100000"
echo "========================================"
echo ""

# Start training with accelerate
echo "Starting multi-GPU training at $(date)"
echo ""

accelerate launch \
    --config_file training/configs/accelerate_single_node.yaml \
    train.py \
    --config configs/base_config.yaml

# Capture exit code
EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code: $EXIT_CODE"
fi
echo "Finished at $(date)"
echo "========================================"

exit $EXIT_CODE
