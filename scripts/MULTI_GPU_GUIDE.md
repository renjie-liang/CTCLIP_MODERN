# Multi-GPU Training Guide

## 📊 重要：Iteration数量变化

使用多GPU会减少每个epoch的iterations，但**总数据量保持不变**：

| 配置 | Batch Size (per GPU) | Total Batch | Steps/Epoch |
|------|---------------------|-------------|-------------|
| 1 GPU | 4 | 4 | 7,375 |
| 2 GPUs | 4 | 8 | 3,687 (50%) |
| 4 GPUs | 4 | 16 | 1,843 (25%) |
| 8 GPUs | 4 | 32 | 921 (12.5%) |

**说明：**
- Steps减少是因为每步处理更多数据
- 每个epoch仍然遍历所有29,500个样本
- 训练时间会减少（并行加速）

---

## 方案1️⃣: 单节点多GPU（最简单）

### 使用场景
- 一台机器上有多个GPU
- 最简单的并行方案
- 推荐先测试这个

### 步骤

**1. 修改配置文件 `accelerate_config_single_node.yaml`**

```yaml
num_processes: 2  # 改成你想用的GPU数量 (2, 4, 8等)
```

**2. 启动训练**

```bash
# 方式A: 使用脚本
bash scripts/train_multi_gpu.sh

# 方式B: 直接命令
accelerate launch \
    --config_file accelerate_config_single_node.yaml \
    train.py \
    --config configs/base_config.yaml
```

**3. 指定特定GPU（可选）**

```bash
# 只使用GPU 0和1
export CUDA_VISIBLE_DEVICES=0,1
bash scripts/train_multi_gpu.sh

# 只使用GPU 2和3
export CUDA_VISIBLE_DEVICES=2,3
bash scripts/train_multi_gpu.sh
```

---

## 方案2️⃣: 多节点多GPU（SLURM）

### 使用场景
- 需要使用多台机器
- 有SLURM作业调度器
- 需要更大规模训练

### 步骤

**1. 修改SLURM脚本 `scripts/train_slurm_multi_node.sh`**

根据你的集群修改：
```bash
#SBATCH --nodes=2                   # 节点数量
#SBATCH --gpus-per-node=4          # 每节点GPU数量
#SBATCH --partition=gpu            # 分区名称
```

**2. 修改配置文件 `accelerate_config_multi_node.yaml`**

```yaml
num_machines: 2      # 节点数量
num_processes: 8     # 总GPU数 = nodes × GPUs per node
```

**3. 提交作业**

```bash
sbatch scripts/train_slurm_multi_node.sh
```

**4. 查看日志**

```bash
# 查看输出
tail -f logs/train_JOBID.out

# 查看错误
tail -f logs/train_JOBID.err
```

---

## ⚙️ 需要调整Learning Rate吗？

当batch size增大时，通常需要调整learning rate：

### Linear Scaling Rule
```
新LR = 原LR × (新batch / 原batch)
```

**示例：**
```yaml
# 原配置 (1 GPU, batch=4)
learning_rate: 1.25e-6

# 2 GPUs (total batch=8)
learning_rate: 2.5e-6  # 1.25e-6 × 2

# 4 GPUs (total batch=16)
learning_rate: 5.0e-6  # 1.25e-6 × 4
```

**但要注意：**
- 对于小batch size (< 256)，可能不需要线性缩放
- 建议先测试原LR，如果不稳定再调整
- 可以配合更长的warmup

---

## 🔍 验证多GPU是否生效

训练开始时会显示：

```
Distributed environment: MULTI_GPU
Number of processes: 2
Number of GPUs: 2
```

使用 `nvidia-smi` 查看GPU使用：
```bash
watch -n 1 nvidia-smi
```

应该看到多个GPU都有显存占用和GPU利用率。

---

## 🐛 常见问题

### 1. 报错：NCCL timeout
**原因：** 节点间网络通信问题

**解决：**
```python
# 在 trainer.py 中已设置超时
init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
```

### 2. 报错：Out of memory
**原因：** 每个GPU仍然加载相同的batch_size

**解决：** 减小配置中的batch_size
```yaml
data:
  batch_size: 2  # 从4减到2
```

### 3. Loss震荡
**原因：** Batch size变大导致训练不稳定

**解决：**
- 增大warmup_steps
- 降低learning rate
- 使用gradient accumulation

---

## 📈 性能对比

预期加速比（理想情况）：

| GPUs | 理论加速 | 实际加速 | 通信开销 |
|------|----------|----------|----------|
| 1 | 1.0x | 1.0x | 0% |
| 2 | 2.0x | 1.8-1.9x | 5-10% |
| 4 | 4.0x | 3.5-3.8x | 5-12% |
| 8 | 8.0x | 6.5-7.0x | 12-18% |

单节点通常比多节点效率更高（通信延迟更低）。

---

## 🎯 建议的训练流程

**第1步：单GPU验证代码** ✅ (你已完成)
```bash
python train.py --config configs/debug_config.yaml
```

**第2步：单节点2 GPU测试**
```bash
# 修改 accelerate_config_single_node.yaml: num_processes: 2
bash scripts/train_multi_gpu.sh
```

**第3步：单节点全部GPU**
```bash
# 修改 accelerate_config_single_node.yaml: num_processes: 4 (或你的GPU数)
bash scripts/train_multi_gpu.sh
```

**第4步：（可选）多节点训练**
```bash
# 修改 SLURM 脚本和配置
sbatch scripts/train_slurm_multi_node.sh
```

---

## 📞 需要帮助？

如果遇到问题，检查：
1. `nvidia-smi` - 确认GPU可见
2. 日志中的 "Number of processes" - 确认GPU数量正确
3. Steps per epoch - 应该减少到原来的 1/N (N=GPU数量)
