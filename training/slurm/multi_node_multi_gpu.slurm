#!/bin/bash
#SBATCH --job-name=ctclip-train
#SBATCH --nodes=2                    # 2个节点
#SBATCH --ntasks-per-node=1         # 每节点1个任务
#SBATCH --gpus-per-node=4           # 每节点4个GPU (根据你的实际情况修改)
#SBATCH --cpus-per-task=32          # 每任务32个CPU (根据你的实际情况修改)
#SBATCH --mem-per-cpu=4G            # 每CPU 4GB内存
#SBATCH --time=48:00:00             # 最长运行时间
#SBATCH --partition=gpu             # GPU分区 (根据你的集群修改)
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# 创建logs目录
mkdir -p logs

# 打印作业信息
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"

# 获取主节点地址
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "Master node: $MASTER_ADDR:$MASTER_PORT"

# 激活环境 (根据你的实际情况修改)
# source /path/to/your/conda/etc/profile.d/conda.sh
# conda activate your_env

# 使用srun + accelerate launch启动多节点训练
srun accelerate launch \
    --config_file training/configs/accelerate_multi_node.yaml \
    --num_processes=$((SLURM_NNODES * SLURM_GPUS_PER_NODE)) \
    --num_machines=$SLURM_NNODES \
    --machine_rank=$SLURM_PROCID \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    train.py \
    --config configs/base_config.yaml

echo "Training completed"
