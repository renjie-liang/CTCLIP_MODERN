# DataLoader 调试指南

当遇到 DataLoader worker 崩溃或加载超时问题时，使用本指南进行诊断。

---

## 常见问题和症状

### 1. Worker 进程退出 (Worker Exited Unexpectedly)

**错误信息**：
```
RuntimeError: DataLoader worker (pid(s) XXXXX) exited unexpectedly
```

**可能原因**：
- 内存不足（3D CT 数据非常大）
- 数据损坏（某些 TAR 文件有问题）
- 进程间通信问题（shared memory 不足）
- 文件描述符耗尽

### 2. 加载超时 (Loading Hangs)

**症状**：
- DataLoader 创建成功，但第一个 batch 永远无法加载
- 进程卡住，没有任何输出

**可能原因**：
- 磁盘 I/O 太慢（/orange 存储可能在高峰期慢）
- Worker 进程死锁
- Shard 文件损坏或格式错误

---

## 调试步骤

### 第 1 步：使用 num_workers=0 测试

**为什么？**
- `num_workers=0` 在主进程中运行，更容易调试
- 可以看到完整的错误堆栈
- 避免多进程的复杂性

**快速测试**：
```bash
cd /orange/xujie/liang.renjie/3DCT/CTCLIP_MODERN

# 基础测试（无详细日志）
python scripts/quick_test_loader.py

# 带详细计时日志
DEBUG_TIMING=true python scripts/quick_test_loader.py
```

**预期结果**：
- ✓ 如果成功：问题在多进程通信
- ✗ 如果失败：问题在数据本身或处理逻辑

---

### 第 2 步：启用详细计时日志

设置环境变量 `DEBUG_TIMING=true` 会打印每个样本的详细处理时间：

```bash
DEBUG_TIMING=true python scripts/quick_test_loader.py
```

**输出示例**：
```
[study_001] Metadata decode: 0.0001s
[study_001] Volume decode: 0.1234s (shape=(1024, 1024, 251))
[study_001] Report decode: 0.0002s
[study_001] Labels decode: 0.0001s
[study_001] Volume process: 2.3456s
[study_001] TOTAL decode time: 2.4694s
```

**分析瓶颈**：
- `Metadata decode` 慢 → JSON 解析问题（不太可能）
- `Volume decode` 慢 → 磁盘 I/O 慢或数据太大
- `Volume process` 慢 → 预处理计算太重（resize/crop/normalize）

---

### 第 3 步：完整诊断测试

运行完整的诊断脚本：

```bash
cd /orange/xujie/liang.renjie/3DCT/CTCLIP_MODERN
python scripts/test_dataloader_debug.py
```

**这个脚本会运行 4 个测试**：
1. **Test 1**: 单样本加载（无 DataLoader）
2. **Test 2**: DataLoader + num_workers=0
3. **Test 3**: DataLoader + num_workers=4
4. **Test 4**: 性能基准测试

**输出解读**：
```
Test Summary
================================================================================
  single_sample       : ✓ PASS
  single_worker       : ✓ PASS
  multi_worker        : ✗ FAIL  ← 问题在这里
  benchmark           : ⚠ SKIP
```

如果 `multi_worker` 失败，说明问题在多进程环境。

---

## 常见问题和解决方案

### 问题 1: 内存不足 (OOM)

**症状**：
- Worker 进程被 kill
- 系统日志显示 OOM (Out of Memory)

**诊断**：
```bash
# 监控内存使用
watch -n 1 free -h

# 或在训练时
htop
```

**解决方案**：
1. **减少 num_workers**：
   ```yaml
   # configs/base_config.yaml
   data:
     num_workers: 16  # 从 32 减少到 16
   ```

2. **减少 prefetch_factor**：
   ```yaml
   data:
     prefetch_factor: 1  # 从 2 减少到 1
   ```

3. **减少 batch_size**：
   ```yaml
   data:
     batch_size: 16  # 从 32 减少到 16
   ```

**公式**：
```
内存使用 ≈ num_workers × prefetch_factor × batch_size × 单样本大小
单样本大小 ≈ 2GB (480×480×240 float32 + 中间数据)

示例：
32 workers × 2 prefetch × 32 batch × 2GB = 4TB！！！（不可能）
16 workers × 1 prefetch × 16 batch × 2GB = 512GB（可能但很紧张）
8 workers × 1 prefetch × 8 batch × 2GB = 128GB（安全）
```

---

### 问题 2: Shared Memory 不足

**症状**：
- `RuntimeError: DataLoader worker exited unexpectedly`
- 系统日志显示 `/dev/shm` 满了

**诊断**：
```bash
df -h /dev/shm
```

**解决方案**：
1. **使用 file_system 共享策略**（训练脚本中添加）：
   ```python
   import torch.multiprocessing as mp
   mp.set_sharing_strategy('file_system')
   ```

2. **增加 shared memory 大小**（需要 root 权限）：
   ```bash
   # 查看当前大小
   df -h /dev/shm

   # 临时增加（需要 sudo）
   sudo mount -o remount,size=128G /dev/shm
   ```

---

### 问题 3: 磁盘 I/O 太慢

**症状**：
- 加载速度极慢（> 5 秒/batch）
- `Volume decode` 时间很长

**诊断**：
```bash
# 测试读取速度
time head -c 1G /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/train_fixed_webdataset/shard-000000.tar > /dev/null
```

**解决方案**：
1. **避免高峰期训练**（白天 I/O 通常更慢）

2. **使用本地缓存**（如果有本地 SSD）：
   ```bash
   # 复制部分数据到本地测试
   cp /orange/.../shard-000000.tar /tmp/
   ```

3. **增加 shuffle_buffer_size**（减少随机读取）：
   ```yaml
   data:
     shuffle_buffer_size: 10000  # 从 1000 增加
   ```

---

### 问题 4: 数据损坏

**症状**：
- 特定样本总是失败
- 错误信息显示解码问题

**诊断**：
```bash
# 测试 TAR 文件完整性
tar -tzf /orange/.../shard-000000.tar | head

# 查看 manifest
cat /orange/.../train_fixed_webdataset/manifest.json
```

**解决方案**：
1. **跳过损坏的样本**（WebDataset 支持）：
   ```python
   dataset = (
       wds.WebDataset(shard_pattern, ...)
       .map(self._decode_sample)
       .select(lambda x: x is not None)  # 过滤损坏样本
   )
   ```

2. **重新生成损坏的 shard**：
   ```bash
   python scripts/convert_npz_to_webdataset.py \
       --shard_index 0 \
       --samples_per_shard 150
   ```

---

## 推荐的训练配置

### 对于调试（快速迭代）

```yaml
# configs/debug_config.yaml
data:
  batch_size: 4         # 小 batch
  num_workers: 0        # 单进程
  prefetch_factor: 2    # 不使用（num_workers=0 时忽略）
```

**启动**：
```bash
DEBUG_TIMING=true python train.py --config configs/debug_config.yaml
```

---

### 对于实际训练（性能优化）

#### 配置 A: 保守（稳定优先）
```yaml
data:
  batch_size: 16
  num_workers: 16
  prefetch_factor: 1
```

**内存估算**：16 × 1 × 16 × 2GB = 512GB

#### 配置 B: 平衡（推荐）
```yaml
data:
  batch_size: 24
  num_workers: 24
  prefetch_factor: 1
```

**内存估算**：24 × 1 × 24 × 2GB = 1.15TB

#### 配置 C: 激进（最快，需要充足内存）
```yaml
data:
  batch_size: 32
  num_workers: 32
  prefetch_factor: 2
```

**内存估算**：32 × 2 × 32 × 2GB = 4TB（可能太大！）

---

## 环境变量

### DEBUG_TIMING

启用详细的样本加载计时：
```bash
DEBUG_TIMING=true python train.py
```

### PYTHONFAULTHANDLER

Python 崩溃时打印堆栈：
```bash
PYTHONFAULTHANDLER=1 python train.py
```

### CUDA_LAUNCH_BLOCKING

CUDA 错误时立即报告（而不是异步）：
```bash
CUDA_LAUNCH_BLOCKING=1 python train.py
```

---

## Slurm 作业中调试

### 在 submit_train.sh 中添加调试选项

```bash
# 启用 Python 错误追踪
export PYTHONFAULTHANDLER=1

# 启用详细计时（仅用于调试）
# export DEBUG_TIMING=true

# 设置共享策略
export TORCH_SHARING_STRATEGY=file_system

# 启动训练
python train.py --config configs/base_config.yaml
```

### 实时查看日志

```bash
# 查看输出
tail -f out_slurm/train_base_<JOB_ID>.out

# 查看错误
tail -f out_slurm/train_base_<JOB_ID>.err

# 同时查看
tail -f out_slurm/train_base_<JOB_ID>.{out,err}
```

---

## 性能优化建议

### 1. 找到最优 num_workers

**方法**：逐步增加并测试
```python
# 脚本：scripts/find_optimal_workers.py
for num_workers in [0, 4, 8, 16, 24, 32]:
    # 测试吞吐量
    # 记录 samples/sec
```

**经验法则**：
- `num_workers = min(cpu_cores - 4, 2 × num_gpus × 8)`
- 40 CPUs → 建议 24-32 workers
- 32 CPUs → 建议 16-24 workers

### 2. 验证数据瓶颈

**GPU 利用率低 (<70%)** → 数据加载是瓶颈
- 增加 num_workers
- 增加 prefetch_factor
- 优化 _process_volume()

**GPU 利用率高 (>90%)** → 计算是瓶颈
- 已经很好！
- 可以减少 num_workers 节省内存

---

## 快速命令参考

```bash
# 快速测试（单进程）
python scripts/quick_test_loader.py

# 带详细日志
DEBUG_TIMING=true python scripts/quick_test_loader.py

# 完整诊断
python scripts/test_dataloader_debug.py

# 监控系统资源
htop
watch -n 1 nvidia-smi

# 检查磁盘 I/O
iostat -x 1

# 检查 shared memory
df -h /dev/shm
```

---

## 总结

### 调试流程

1. ✅ 首先用 `num_workers=0` 测试 → 排除数据问题
2. ✅ 启用 `DEBUG_TIMING=true` → 找到瓶颈
3. ✅ 逐步增加 num_workers → 找到最优配置
4. ✅ 监控内存和 GPU 利用率 → 验证平衡

### 关键原则

- **调试时**：num_workers=0, batch_size=4, DEBUG_TIMING=true
- **训练时**：根据内存调整 num_workers 和 batch_size
- **优化时**：监控 GPU 利用率，确保 >85%

---

**如有问题，按顺序检查**：
1. 数据加载（quick_test_loader.py）
2. 内存使用（htop）
3. 磁盘 I/O（iostat）
4. GPU 利用率（nvidia-smi）
