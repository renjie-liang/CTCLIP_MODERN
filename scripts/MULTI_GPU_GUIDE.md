# Multi-GPU Training Guide

## 📊 Important: Changes in Iteration Count

Using multiple GPUs reduces iterations per epoch, but **total data volume remains unchanged**:

| Configuration | Batch Size (per GPU) | Total Batch | Steps/Epoch |
|------|---------------------|-------------|-------------|
| 1 GPU | 4 | 4 | 7,375 |
| 2 GPUs | 4 | 8 | 3,687 (50%) |
| 4 GPUs | 4 | 16 | 1,843 (25%) |
| 8 GPUs | 4 | 32 | 921 (12.5%) |

**Explanation:**
- Steps decrease because each step processes more data
- Each epoch still traverses all 29,500 samples
- Training time will decrease (parallel acceleration)

---

## Solution 1️⃣: Single-Node Multi-GPU (Easiest)

### Use Cases
- Multiple GPUs on a single machine
- Simplest parallel solution
- Recommended to test this first

### Steps

**1. Modify the config file `accelerate_config_single_node.yaml`**

```yaml
num_processes: 2  # Change to the number of GPUs you want to use (2, 4, 8, etc.)
```

**2. Start Training**

```bash
# Method A: Using script
bash scripts/train_multi_gpu.sh

# Method B: Direct command
accelerate launch \
    --config_file accelerate_config_single_node.yaml \
    train.py \
    --config configs/base_config.yaml
```

**3. Specify Specific GPUs (Optional)**

```bash
# Use only GPU 0 and 1
export CUDA_VISIBLE_DEVICES=0,1
bash scripts/train_multi_gpu.sh

# Use only GPU 2 and 3
export CUDA_VISIBLE_DEVICES=2,3
bash scripts/train_multi_gpu.sh
```

---

## Solution 2️⃣: Multi-Node Multi-GPU (SLURM)

### Use Cases
- Need to use multiple machines
- Have SLURM job scheduler
- Need larger scale training

### Steps

**1. Modify SLURM script `scripts/train_slurm_multi_node.sh`**

Modify according to your cluster:
```bash
#SBATCH --nodes=2                   # Number of nodes
#SBATCH --gpus-per-node=4          # Number of GPUs per node
#SBATCH --partition=gpu            # Partition name
```

**2. Modify config file `accelerate_config_multi_node.yaml`**

```yaml
num_machines: 2      # Number of nodes
num_processes: 8     # Total GPUs = nodes × GPUs per node
```

**3. Submit Job**

```bash
sbatch scripts/train_slurm_multi_node.sh
```

**4. View Logs**

```bash
# View output
tail -f logs/train_JOBID.out

# View errors
tail -f logs/train_JOBID.err
```

---

## ⚙️ Do You Need to Adjust Learning Rate?

When batch size increases, usually you need to adjust the learning rate:

### Linear Scaling Rule
```
New LR = Original LR × (New batch / Original batch)
```

**Example:**
```yaml
# Original config (1 GPU, batch=4)
learning_rate: 1.25e-6

# 2 GPUs (total batch=8)
learning_rate: 2.5e-6  # 1.25e-6 × 2

# 4 GPUs (total batch=16)
learning_rate: 5.0e-6  # 1.25e-6 × 4
```

**But note:**
- For small batch sizes (< 256), linear scaling may not be necessary
- Recommend testing with original LR first, adjust if unstable
- Can be combined with longer warmup

---

## 🔍 Verify Multi-GPU is Working

At training start, you will see:

```
Distributed environment: MULTI_GPU
Number of processes: 2
Number of GPUs: 2
```

Use `nvidia-smi` to check GPU usage:
```bash
watch -n 1 nvidia-smi
```

You should see multiple GPUs with memory usage and GPU utilization.

---

## 🐛 Common Issues

### 1. Error: NCCL timeout
**Cause:** Network communication issues between nodes

**Solution:**
```python
# Timeout already set in trainer.py
init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
```

### 2. Error: Out of memory
**Cause:** Each GPU still loads the same batch_size

**Solution:** Reduce batch_size in config
```yaml
data:
  batch_size: 2  # Reduce from 4 to 2
```

### 3. Loss Oscillation
**Cause:** Larger batch size causes training instability

**Solution:**
- Increase warmup_steps
- Lower learning rate
- Use gradient accumulation

---

## 📈 Performance Comparison

Expected speedup (ideal case):

| GPUs | Theoretical Speedup | Actual Speedup | Communication Overhead |
|------|----------|----------|----------|
| 1 | 1.0x | 1.0x | 0% |
| 2 | 2.0x | 1.8-1.9x | 5-10% |
| 4 | 4.0x | 3.5-3.8x | 5-12% |
| 8 | 8.0x | 6.5-7.0x | 12-18% |

Single-node is typically more efficient than multi-node (lower communication latency).

---

## 🎯 Recommended Training Workflow

**Step 1: Single GPU code verification** ✅ (You've completed this)
```bash
python train.py --config configs/debug_config.yaml
```

**Step 2: Single-node 2 GPU test**
```bash
# Modify accelerate_config_single_node.yaml: num_processes: 2
bash scripts/train_multi_gpu.sh
```

**Step 3: Single-node all GPUs**
```bash
# Modify accelerate_config_single_node.yaml: num_processes: 4 (or your GPU count)
bash scripts/train_multi_gpu.sh
```

**Step 4: (Optional) Multi-node training**
```bash
# Modify SLURM script and config
sbatch scripts/train_slurm_multi_node.sh
```

---

## 📞 Need Help?

If you encounter problems, check:
1. `nvidia-smi` - Confirm GPUs are visible
2. "Number of processes" in logs - Confirm GPU count is correct
3. Steps per epoch - Should be reduced to 1/N of original (N=number of GPUs)
