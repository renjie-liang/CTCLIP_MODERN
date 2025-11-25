#!/bin/bash

#SBATCH --job-name=ctclip_resume_training
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:2              # Use 2 GPUs (can change to 4/8 etc.)
#SBATCH --nodes=1                 # Single node
#SBATCH --cpus-per-task=64        # Recommended to increase CPUs to match GPU count
#SBATCH --mem=200gb               # Recommended to increase memory to match GPU count
#SBATCH --time=72:00:00
#SBATCH --account=xujie
#SBATCH --qos=xujie
#SBATCH --output=out_slurm/resume_training_%j.out
#SBATCH --error=out_slurm/resume_training_%j.err

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
echo "GPUs: 2 (CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES)"
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
echo "Resume Information"
echo "========================================"
echo "Resuming from checkpoint: saves/baseline/latest.pt"
echo ""

# Start training with accelerate - RESUME FROM CHECKPOINT
echo "Starting resumed multi-GPU training at $(date)"
echo ""

accelerate launch \
    --config_file run_setup/configs/accelerate_single_node.yaml \
    train.py \
    --config configs/base_config.yaml \
    --resume saves/baseline/latest.pt

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
