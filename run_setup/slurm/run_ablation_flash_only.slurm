#!/bin/bash

#SBATCH --job-name=ctclip_flash_only
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200gb
#SBATCH --time=72:00:00
#SBATCH --account=xujie
#SBATCH --qos=xujie
#SBATCH --output=out_slurm/ablation_flash_only_%j.out
#SBATCH --error=out_slurm/ablation_flash_only_%j.err

mkdir -p out_slurm logs saves/ablation_flash_only

export MAMBA_EXE='/home/liang.renjie/micromamba'
export MAMBA_ROOT_PREFIX='/blue/xujie/liang.renjie/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate b200

cd /orange/xujie/liang.renjie/3DCT/CTCLIP_MODERN

echo "========================================"
echo "Experiment: FlashAttention ONLY"
echo "Config: configs/experiments/ablation_flash_only.yaml"
echo "FlashAttention: ON | RMSNorm: OFF | SwiGLU: OFF"
echo "========================================"

accelerate launch \
    --config_file run_setup/configs/accelerate_single_node.yaml \
    train.py \
    --config configs/experiments/ablation_flash_only.yaml

EXIT_CODE=$?
echo "Finished at $(date) with exit code: $EXIT_CODE"
exit $EXIT_CODE
