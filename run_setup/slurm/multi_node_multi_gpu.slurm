#!/bin/bash
#SBATCH --job-name=ctclip-train
#SBATCH --nodes=2                    # 2 nodes
#SBATCH --ntasks-per-node=1         # 1 task per node
#SBATCH --gpus-per-node=4           # 4 GPUs per node (modify according to your setup)
#SBATCH --cpus-per-task=32          # 32 CPUs per task (modify according to your setup)
#SBATCH --mem-per-cpu=4G            # 4GB memory per CPU
#SBATCH --time=48:00:00             # Maximum runtime
#SBATCH --partition=gpu             # GPU partition (modify for your cluster)
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Create logs directory
mkdir -p logs

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"

# Get master node address
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "Master node: $MASTER_ADDR:$MASTER_PORT"

# Activate environment (modify according to your setup)
# source /path/to/your/conda/etc/profile.d/conda.sh
# conda activate your_env

# Launch multi-node training with srun + accelerate
srun accelerate launch \
    --config_file run_setup/configs/accelerate_multi_node.yaml \
    --num_processes=$((SLURM_NNODES * SLURM_GPUS_PER_NODE)) \
    --num_machines=$SLURM_NNODES \
    --machine_rank=$SLURM_PROCID \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    train.py \
    --config configs/base_config.yaml

echo "Training completed"
