# Training Scripts and Configurations

这个文件夹包含所有与训练相关的配置文件和提交脚本。

## 文件夹结构

```
training/
├── slurm/                          # SLURM 集群提交脚本
│   ├── single_gpu.slurm            # 单节点单GPU训练
│   ├── single_node_multi_gpu.slurm # 单节点多GPU训练
│   └── multi_node_multi_gpu.slurm  # 多节点多GPU训练
├── bash/                           # 非SLURM的训练脚本
│   └── train_single_node_multi_gpu.sh  # 单节点多GPU训练（交互式/开发环境）
└── configs/                        # Accelerate配置文件
    ├── accelerate_single_node.yaml # 单节点多GPU的accelerate配置
    └── accelerate_multi_node.yaml  # 多节点多GPU的accelerate配置
```

## 使用说明

### 1. 单节点单GPU训练 (SLURM)

适用场景：快速测试、小规模训练

```bash
sbatch training/slurm/single_gpu.slurm
```

配置：
- 1个GPU
- Batch size: 8
- 不使用accelerate
- 内存: 200GB

### 2. 单节点多GPU训练 (SLURM)

适用场景：中等规模训练、单机多卡加速

```bash
sbatch training/slurm/single_node_multi_gpu.slurm
```

配置：
- 默认4个GPU (可修改 `#SBATCH --gres=gpu:4`)
- 使用accelerate进行分布式训练
- Batch size per GPU: 8
- 有效 batch size: 32 (8 × 4)
- 内存: 400GB

**修改GPU数量：**
1. 修改 `single_node_multi_gpu.slurm` 中的 `#SBATCH --gres=gpu:N`
2. 修改 `configs/accelerate_single_node.yaml` 中的 `num_processes: N`

### 3. 多节点多GPU训练 (SLURM)

适用场景：大规模训练、需要多台机器

```bash
sbatch training/slurm/multi_node_multi_gpu.slurm
```

配置：
- 默认2个节点，每节点4个GPU (总共8个GPU)
- 使用accelerate + srun进行分布式训练
- 需要修改分区、账户等信息以适配你的集群

**修改节点/GPU配置：**
1. 修改 `multi_node_multi_gpu.slurm` 中的 `#SBATCH --nodes` 和 `#SBATCH --gpus-per-node`
2. 修改 `configs/accelerate_multi_node.yaml` 中的 `num_machines` 和 `num_processes`

### 4. 单节点多GPU训练 (非SLURM)

适用场景：交互式训练、开发环境、非SLURM集群

```bash
bash training/bash/train_single_node_multi_gpu.sh
```

或者直接使用accelerate命令：

```bash
accelerate launch \
    --config_file training/configs/accelerate_single_node.yaml \
    train.py \
    --config configs/base_config.yaml
```

## Accelerate配置说明

### accelerate_single_node.yaml

单节点多GPU配置：
- `num_processes`: GPU数量（默认2）
- `mixed_precision`: fp16混合精度训练
- `machine_rank`: 0（单机）

### accelerate_multi_node.yaml

多节点多GPU配置：
- `num_machines`: 节点数量（默认2）
- `num_processes`: 总GPU数量（默认8）
- `machine_rank`: 会被环境变量覆盖
- `main_process_ip`: 会被SLURM脚本设置

## 注意事项

1. **环境配置**：所有SLURM脚本都假设使用micromamba，如需修改请编辑相应的环境激活部分

2. **路径配置**：SLURM脚本中包含特定的项目路径，使用前需要修改：
   - `MAMBA_EXE`
   - `MAMBA_ROOT_PREFIX`
   - 项目目录路径

3. **集群特定配置**：
   - `--partition`: 分区名称
   - `--account`: 账户名称
   - `--qos`: QOS名称
   - 根据你的集群配置修改这些参数

4. **输出目录**：训练日志会保存在：
   - SLURM输出: `out_slurm/`
   - 训练日志: `logs/`
   - 模型保存: `saves/`

5. **Batch Size计算**：
   - 单GPU: batch_size = 8
   - 多GPU: 有效 batch_size = batch_size × num_gpus
   - 例如：4个GPU时，有效batch_size = 8 × 4 = 32

## 故障排查

如果训练失败，请检查：
1. SLURM输出文件 (`out_slurm/train_*.err`)
2. GPU是否正常分配 (检查 `CUDA_VISIBLE_DEVICES`)
3. Accelerate配置中的GPU数量是否匹配SLURM申请的GPU数量
4. 内存是否足够 (多GPU训练可能需要更多内存)
5. WebDataset路径是否正确
