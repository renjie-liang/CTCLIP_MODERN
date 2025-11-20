#!/bin/bash

#SBATCH --job-name=ctclip_train
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=200gb
#SBATCH --time=72:00:00
#SBATCH --account=xujie
#SBATCH --qos=xujie
#SBATCH --output=out_slurm/train_base_%j.out
#SBATCH --error=out_slurm/train_base_%j.err

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
echo "Memory: 200GB"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "========================================"
echo "Environment"
echo "========================================"
echo "Working directory: $(pwd)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "========================================"
echo "Training Configuration"
echo "========================================"
echo "Config file: configs/base_config.yaml"
echo "Data format: WebDataset (float16)"
echo "Mixed precision: Enabled (fp16)"
echo "Batch size: 8"
echo "Num workers: 24"
echo "Max steps: 100000"
echo "========================================"
echo ""

# Start training
echo "Starting training at $(date)"
echo ""

python train.py --config configs/base_config.yaml

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
