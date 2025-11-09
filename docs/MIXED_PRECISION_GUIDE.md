# 混合精度训练（Mixed Precision Training）配置指南

## 概述

本项目采用**三层 float16 优化策略**，在存储、加载和训练各阶段平衡了性能、精度和稳定性。

---

## 完整数据流和 dtype 策略

### 1. 存储层（Storage）- float16

**位置**: WebDataset TAR 文件
**格式**: 原始二进制字节流（`.bin` 扩展名）
**Dtype**: `float16`

```python
# scripts/convert_npz_to_webdataset.py
volume_fp16 = np.ascontiguousarray(npz_data.astype(np.float16))
volume_bytes = volume_fp16.tobytes()  # 原始字节，不使用 pickle
```

**优点**:
- ✅ 存储空间压缩（相比 float32 理论上节省 50%）
- ✅ 本项目实际压缩比：float64 → float16 约 4x
- ✅ 实际数据：14TB → 11.5TB（1.18x，因为原始 NPZ 已有 zip 压缩）

**为什么选择 float16？**
- CT 扫描的 Hounsfield Units (HU) 范围：-1000 到 +1000
- float16 动态范围：-65504 到 +65504，**完全足够**
- 精度损失：< 0.1%，对医学图像分析**可接受**

---

### 2. 加载层（Loading）- float16 → float32

**位置**: `src/data/webdataset_loader.py`
**转换点**: `_process_volume()` 函数第一步

```python
def _process_volume(self, volume_data: np.ndarray, metadata: dict) -> torch.Tensor:
    # 立即从 float16 转换为 float32
    img_data = volume_data.astype(np.float32)

    # 后续所有处理都用 float32
    img_data = slope * img_data + intercept
    img_data = np.clip(img_data, hu_min, hu_max)
    tensor = torch.tensor(img_data, dtype=torch.float32)
    ...
```

**为什么转换为 float32？**
- ✅ **数值稳定性**: 预处理涉及多次数学运算（rescale、clip、normalize）
- ✅ **中间精度**: 避免多次 float16 运算累积误差
- ✅ **兼容性**: PyTorch 默认 dtype 是 float32，避免类型不匹配

**性能影响**:
- 转换开销：可忽略（<1ms per sample）
- 内存占用：预处理阶段短暂增加，训练时会被 AMP 优化回来

---

### 3. 训练层（Training）- 混合精度（AMP）

**位置**: `src/training/trainer.py`
**实现**: HuggingFace Accelerate 的自动混合精度

#### 3.1 Accelerator 初始化

```python
self.accelerator = Accelerator(
    kwargs_handlers=[ddp_kwargs, init_kwargs],
    mixed_precision='fp16'  # 启用自动混合精度
)
```

#### 3.2 前向传播

```python
# 输入是 float32 tensor
with self.accelerator.autocast():
    # autocast 会自动选择 float16/float32
    # - 矩阵乘法：float16（利用 Tensor Cores）
    # - LayerNorm/Softmax：float32（数值稳定）
    loss = self.model(text_tokens, volume_tensor, return_loss=True, device=device)
```

#### 3.3 反向传播

```python
# Accelerator 自动处理梯度缩放（gradient scaling）
self.accelerator.backward(loss)
```

**混合精度（AMP）的工作原理**:

| 操作类型 | 使用精度 | 原因 |
|---------|---------|------|
| 矩阵乘法（Linear、Conv） | float16 | Tensor Core 加速，速度提升 2-3x |
| 激活函数（ReLU、GELU） | float16 | 简单操作，float16 足够 |
| LayerNorm、BatchNorm | float32 | 需要高精度统计量 |
| Softmax、LogSoftmax | float32 | 避免数值溢出 |
| Loss 计算 | float32 | 梯度计算需要高精度 |
| 梯度累积 | float32 | 避免梯度下溢（underflow） |
| 优化器更新 | float32 | 权重更新需要高精度 |

**梯度缩放（Gradient Scaling）**:
- 问题：float16 最小正数 ≈ 6e-8，梯度可能下溢
- 解决：
  1. 前向传播：正常计算
  2. 反向传播前：Loss 乘以缩放因子（如 2^16）
  3. 梯度计算：缩放后的梯度（避免下溢）
  4. 优化器更新前：梯度除以缩放因子（恢复原值）
  5. 权重更新：float32 精度更新

Accelerate 的 `autocast()` 和 `backward()` **自动处理所有这些细节**！

---

## 性能预期

### 训练速度提升

| GPU 型号 | 预期加速比 | Tensor Core |
|---------|----------|-------------|
| V100 | 1.5 - 2.0x | 是 |
| A100 | 2.0 - 3.0x | 是（第三代） |
| H100 | 2.5 - 3.5x | 是（第四代） |
| RTX 3090 | 1.8 - 2.5x | 是 |
| RTX 4090 | 2.2 - 3.0x | 是（第四代） |
| T4 | 1.3 - 1.8x | 是 |

**实际加速取决于**:
- 模型架构（矩阵乘法占比）
- Batch size（越大加速越明显）
- 数据加载速度（避免成为瓶颈）

### 显存节省

- **模型权重**: 减少 ~50%（float32 → float16）
- **激活值**: 减少 ~50%（大部分 float16）
- **梯度**: 保持 float32（为了稳定性）
- **总体**: 通常节省 30-40% 显存

**实际意义**:
- 可以增大 batch size（提升训练效率）
- 或训练更大的模型

### 精度损失

**对于 CLIP 模型（大量研究验证）**:
- AUROC 变化：< 0.1%
- F1 Score 变化：< 0.2%
- 训练稳定性：与 float32 几乎相同

**医学图像任务特点**:
- 输入范围有限（-1 到 +1，已归一化）
- float16 精度完全足够表示
- 临床指标（AUROC/AUPRC）对小的数值误差不敏感

---

## 完整流程总结

```
磁盘（14TB float64 NPZ）
   ↓ [convert_npz_to_webdataset.py]
WebDataset (11.5TB float16 TAR)
   ↓ [webdataset_loader.py: _decode_sample()]
内存（float16 raw bytes）
   ↓ [webdataset_loader.py: _process_volume() line 113]
内存（float32 numpy）
   ↓ [预处理：rescale, clip, resize, normalize]
内存（float32 torch.Tensor）
   ↓ [trainer.py: forward pass]
GPU（autocast: 自动 float16/float32 混合）
   ↓ [model: CTViT + BiomedBERT]
Loss（float32）
   ↓ [backward: gradient scaling]
梯度（float32）
   ↓ [optimizer: AdamW]
权重更新（float32）
```

---

## 验证和监控

### 训练时检查（Training Logs）

启动训练后，检查日志中是否有：

```
Accelerator(Device='cuda', fp16=True, ...)
```

如果看到 `fp16=True`，说明混合精度**已启用** ✓

### 显存占用对比

**预期变化**（相比纯 float32）:

```bash
# 混合精度前（float32）
nvidia-smi  # 显存占用：例如 20GB

# 混合精度后（AMP）
nvidia-smi  # 显存占用：例如 13-15GB（节省 30-40%）
```

### 训练速度对比

**预期变化**（相比纯 float32）:

- 每个 step 的时间：减少 30-60%
- 吞吐量（samples/sec）：提升 1.5-3.0x

### 精度监控

**重要指标**:
- 训练 loss：应该正常下降，不应震荡
- 验证 AUROC：与 float32 基线相差 < 0.5%
- 如果 loss 出现 NaN：
  - 检查学习率是否过大
  - 检查梯度裁剪（`max_grad_norm`）

---

## 故障排查

### 问题 1: Loss 变成 NaN

**原因**: 学习率过大 + float16 数值范围有限

**解决**:
1. 降低学习率（减半试试）
2. 增强梯度裁剪：`max_grad_norm: 0.5 → 0.3`
3. 检查数据预处理是否有异常值

### 问题 2: 训练不稳定

**原因**: 某些操作在 float16 下精度不足

**解决**:
- 切换到 `bf16`（bfloat16）：
  ```python
  mixed_precision='bf16'  # 需要 A100/H100
  ```
- bfloat16 优点：动态范围更大，更稳定
- bfloat16 缺点：需要较新 GPU

### 问题 3: 没有加速

**可能原因**:
1. GPU 不支持 Tensor Cores（检查 GPU 型号）
2. Batch size 过小（增大到 32 以上）
3. 数据加载成为瓶颈（增加 `num_workers`）

**验证**:
```bash
# 检查 GPU 是否支持 float16 加速
nvidia-smi --query-gpu=compute_cap --format=csv
# Compute Capability >= 7.0 支持 Tensor Cores
```

---

## 推荐配置

### 对于本项目（CT-CLIP）

**最佳实践**:
- ✅ **存储**: float16 WebDataset（已完成）
- ✅ **加载**: float16 → float32 预处理（已完成）
- ✅ **训练**: mixed_precision='fp16'（已启用）
- ✅ **Batch size**: 32（充分利用 Tensor Cores）
- ✅ **梯度裁剪**: max_grad_norm=0.5（防止梯度爆炸）

**配置文件**:
```yaml
# configs/base_config.yaml
data:
  batch_size: 32
  num_workers: 32

training:
  max_grad_norm: 0.5
  learning_rate: 1.25e-6
```

---

## 参考资料

- [PyTorch AMP 官方文档](https://pytorch.org/docs/stable/amp.html)
- [HuggingFace Accelerate Mixed Precision](https://huggingface.co/docs/accelerate/usage_guides/mixed_precision)
- [NVIDIA Tensor Cores](https://www.nvidia.com/en-us/data-center/tensor-cores/)
- [Mixed Precision Training (ICLR 2018)](https://arxiv.org/abs/1710.03740)

---

## 总结

| 阶段 | Dtype | 目的 | 性能影响 |
|-----|-------|------|---------|
| 磁盘存储 | float16 | 减少存储空间（14TB → 11.5TB） | 1.18x 压缩 |
| 数据加载 | float32 | 预处理数值稳定性 | 可忽略 |
| 模型训练 | 混合精度 | 加速训练 + 节省显存 | 1.5-3x 加速 |

**结论**: 本项目的三层优化策略是**最佳实践**，既保证了精度，又最大化了性能！✓
