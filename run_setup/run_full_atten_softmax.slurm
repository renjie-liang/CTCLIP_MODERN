#!/bin/bash

#SBATCH --job-name=ctclip_baseline
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1              
#SBATCH --nodes=1                 # Single node
#SBATCH --cpus-per-task=32
#SBATCH --mem=100gb
#SBATCH --time=128:00:00
#SBATCH --account=xujie
#SBATCH --qos=xujie
#SBATCH --output=out_slurm/baseline_%j.out
#SBATCH --error=out_slurm/baseline_%j.err

# Create output directories
mkdir -p out_slurm
mkdir -p logs
mkdir -p saves/baseline

# Setup Micromamba environment
export MAMBA_EXE='/home/liang.renjie/micromamba'
export MAMBA_ROOT_PREFIX='/blue/xujie/liang.renjie/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate b200

# Change to project directory
cd /orange/xujie/liang.renjie/3DCT/CTCLIP_MODERN

python train.py  --config configs/atten_softmax.yaml
