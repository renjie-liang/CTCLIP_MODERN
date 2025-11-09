# Slurm 作业提交指南

本指南介绍如何在 HiPerGator B200 集群上提交 CT-CLIP 训练任务。

---

## 📋 可用的提交脚本

项目提供了两个 Slurm 提交脚本，根据可用的 CPU 资源选择：

### 1. `submit_train.sh` - 推荐配置 ✅

**资源配置**：
- GPU: 1x B200
- CPU: 40 核
- 内存: 200GB
- 时间: 72 小时
- DataLoader workers: 32

**适用场景**：
- 集群有充足的 CPU 资源
- 追求最快的训练速度
- GPU 利用率最大化

**配置文件**：`configs/base_config.yaml`

### 2. `submit_train_reduced_cpu.sh` - 备选配置

**资源配置**：
- GPU: 1x B200
- CPU: 32 核
- 内存: 200GB
- 时间: 72 小时
- DataLoader workers: 24

**适用场景**：
- 集群 CPU 资源受限
- 40 核申请困难或等待时间长
- 可接受略慢的数据加载速度

**配置文件**：`configs/base_config_reduced_cpu.yaml`

---

## 🚀 快速开始

### 步骤 1: 检查数据

在提交作业前，先验证数据路径是否正确：

```bash
# SSH 到服务器
ssh <your-server>

# 检查训练数据
ls /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/train_fixed_webdataset/ | head -5
# 应该看到: shard-000000.tar, shard-000001.tar, ...

# 检查验证数据
ls /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/valid_fixed_webdataset/ | head -5
# 应该看到: shard-000000.tar, shard-000001.tar, ...

# 检查元数据
ls /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/radiology_text_reports/
ls /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/metadata/
ls /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/multi_abnormality_labels/
```

### 步骤 2: 进入项目目录

```bash
cd /orange/xujie/liang.renjie/3DCT/CTCLIP_MODERN
```

### 步骤 3: 提交作业

**方案 A: 使用推荐配置（40 CPUs）**
```bash
sbatch submit_train.sh
```

**方案 B: 使用减少 CPU 配置（32 CPUs）**
```bash
sbatch submit_train_reduced_cpu.sh
```

### 步骤 4: 查看作业状态

```bash
# 查看队列中的作业
squeue -u liang.renjie

# 输出示例：
#   JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
#  123456 hpg-b200  ctclip_t liang.re  R       1:23      1 c0123a-s45
```

---

## 📊 监控训练

### 实时查看训练日志

```bash
# 查看标准输出（训练进度）
tail -f out_slurm/train_base_<JOB_ID>.out

# 查看错误输出（如果有问题）
tail -f out_slurm/train_base_<JOB_ID>.err
```

### 关键日志检查点

#### 1. 环境信息（作业开始时）
```
========================================
Job Information
========================================
Job ID: 123456
Node: c0123a-s45
CPUs per task: 40
GPUs: 0
========================================
Environment
========================================
Python version: Python 3.x.x
PyTorch version: 2.x.x
CUDA available: True
GPU name: NVIDIA B200
========================================
```

#### 2. 混合精度检查
```
Accelerator(Device='cuda:0', fp16=True, ...)
```
**重要**：确保看到 `fp16=True`，表示混合精度已启用！

#### 3. 训练进度
```
Step 100/10000 | Loss: 0.1234 | LR: 1.25e-6 | GPU Mem: 45GB | Time: 1.23s/step
Step 200/10000 | Loss: 0.1156 | LR: 1.30e-6 | GPU Mem: 45GB | Time: 1.20s/step
...
```

#### 4. 验证结果（每 1000 steps）
```
Validation Step 1000:
  AUROC: 0.7234
  AUPRC: 0.6891
  F1: 0.5432
  Precision: 0.5123
  Recall: 0.5789
```

### SSH 到计算节点监控

```bash
# 获取作业运行的节点名称
squeue -u liang.renjie

# SSH 到该节点（例如 c0123a-s45）
ssh c0123a-s45

# 监控 GPU 利用率
watch -n 1 nvidia-smi

# 监控 CPU 和内存
htop
```

**GPU 监控关键指标**：
- **GPU-Util**: 应该 > 85%（如果 < 70%，数据加载可能是瓶颈）
- **Memory-Usage**:
  - 混合精度（AMP）: 约 40-60GB
  - 纯 float32: 约 70-90GB
  - 如果看到显存使用明显减少，说明 AMP 已生效

**CPU 监控关键指标**：
- **40 CPUs 配置**: 应该看到 30-35 个 CPU 核心被使用
- **32 CPUs 配置**: 应该看到 25-28 个 CPU 核心被使用

---

## 📁 输出文件位置

### 训练日志
```
out_slurm/
├── train_base_123456.out  # 标准输出
└── train_base_123456.err  # 错误输出
```

### Checkpoint
```
saves/
├── checkpoint_step_1000.pt
├── checkpoint_step_2000.pt
├── checkpoint_step_3000.pt
├── ...
└── best_model.pt  # 最佳模型（基于 validation AUROC）
```

### WandB 日志

如果启用了 WandB（默认启用），可以在浏览器中查看：
```
https://wandb.ai/<your-entity>/ct-clip
```

---

## ⏱️ 预估训练时间

### 基于 B200 GPU + 混合精度（AMP）

**每个 step 的时间**：
- 预期: 1.0 - 2.0 秒/step
- 取决于: 数据加载速度、GPU 利用率

**总训练时间（10,000 steps）**：
```
最佳情况: 10,000 × 1.0s = 2.8 小时
典型情况: 10,000 × 1.5s = 4.2 小时
保守估计: 10,000 × 2.0s = 5.6 小时

加上验证时间（10 次 × 5 分钟）: + 50 分钟
加上保存 checkpoint（10 次 × 1 分钟）: + 10 分钟

总计: 约 5-7 小时
```

**72 小时的时间限制非常充足！**

---

## 🛠️ 故障排查

### 问题 1: 作业一直在队列中（PD 状态）

```bash
squeue -u liang.renjie
# JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
# 123456 hpg-b200  ctclip_t liang.re PD       0:00      1 (Resources)
```

**原因**：
- 等待 GPU 资源
- 等待 CPU 资源（40 核可能在高峰期难以分配）

**解决方案**：
1. 耐心等待（通常几分钟到几小时）
2. 如果等待时间过长，使用 `submit_train_reduced_cpu.sh`（32 核更容易分配）
3. 检查 QOS 限制：`sacctmgr show qos xujie`

### 问题 2: 作业立即失败（出现在队列后马上消失）

```bash
# 查看作业历史
sacct -j <JOB_ID>

# 查看错误日志
cat out_slurm/train_base_<JOB_ID>.err
```

**常见原因**：
1. **环境激活失败**：检查 micromamba 路径
2. **导入错误**：缺少 Python 包
3. **数据路径错误**：检查 WebDataset 路径

### 问题 3: 训练过程中 Loss 变成 NaN

**查看日志**：
```bash
grep -i "nan\|inf" out_slurm/train_base_<JOB_ID>.out
```

**可能原因**：
- 学习率过大（不太可能，当前 1.25e-6 很保守）
- 混合精度数值不稳定（罕见）

**解决方案**：
1. 降低学习率：`learning_rate: 1.25e-6 → 5e-7`
2. 增强梯度裁剪：`max_grad_norm: 0.5 → 0.3`
3. 切换到 bfloat16（需要修改代码，见下文）

### 问题 4: GPU 利用率低（< 70%）

**原因**：数据加载速度跟不上 GPU 计算速度

**解决方案**：
1. 增加 CPU 资源（切换到 40 核配置）
2. 增加 num_workers（但不超过 CPU 核心数 - 5）
3. 检查存储 IO 性能（`/orange` 可能在高峰期慢）

### 问题 5: Out of Memory (OOM)

**错误信息**：
```
RuntimeError: CUDA out of memory
```

**解决方案**：
1. 减少 batch size：`batch_size: 32 → 24 或 16`
2. 确认混合精度已启用（查看日志中的 `fp16=True`）
3. 减少 prefetch_factor：`prefetch_factor: 2 → 1`

---

## 🔧 高级配置

### 切换到 bfloat16（可选）

如果 float16 训练不稳定，B200 GPU 原生支持 bfloat16：

```python
# 编辑 src/training/trainer.py:95
# 从:
mixed_precision='fp16'
# 改为:
mixed_precision='bf16'
```

**bfloat16 优点**：
- 动态范围更大（与 float32 相同）
- 数值稳定性更好
- B200/H100 原生硬件支持

### 调整验证频率

如果想更频繁地查看验证结果：

```yaml
# configs/base_config.yaml
validation:
  eval_every_n_steps: 500  # 从 1000 改为 500
  eval_samples: 200
```

### 增加保存频率

```yaml
# configs/base_config.yaml
training:
  save_every_n_steps: 500  # 从 1000 改为 500
```

---

## 📞 取消/暂停作业

### 取消作业

```bash
# 取消特定作业
scancel <JOB_ID>

# 取消所有自己的作业
scancel -u liang.renjie
```

### 暂停作业（不推荐）

```bash
# 暂停
scontrol hold <JOB_ID>

# 恢复
scontrol release <JOB_ID>
```

---

## ✅ 检查清单

提交作业前，请确认：

- [ ] 数据路径存在且可访问
- [ ] Micromamba 环境路径正确
- [ ] 输出目录已创建或脚本会自动创建
- [ ] 有足够的磁盘空间（checkpoint 需要 ~5GB/次，共 10 次 = 50GB）
- [ ] WandB 登录（如果使用 WandB）：`wandb login`
- [ ] 混合精度已启用（检查代码 `src/training/trainer.py:95`）

---

## 📚 相关文档

- [混合精度训练指南](docs/MIXED_PRECISION_GUIDE.md)
- [WebDataset 转换指南](docs/WEBDATASET_GUIDE.md)
- [完整数据转换指南](docs/FULL_CONVERSION_GUIDE.md)

---

## 🆘 获取帮助

如果遇到问题：

1. 检查错误日志：`cat out_slurm/train_base_<JOB_ID>.err`
2. 查看完整输出：`cat out_slurm/train_base_<JOB_ID>.out`
3. 检查集群状态：`sinfo -p hpg-b200`
4. 查看账户限制：`sacctmgr show qos xujie`

---

**祝训练顺利！🚀**
