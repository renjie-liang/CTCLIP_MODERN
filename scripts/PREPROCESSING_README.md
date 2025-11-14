# CT-RATE 数据预处理指南

本指南介绍如何从 Hugging Face 直接构建预处理后的 WebDataset，以实现 **10x 训练加速**。

## 🎯 目标

- 将 CPU 密集型预处理（resize, normalize 等）提前完成
- 训练时只需快速读取预处理好的数据
- 数据加载从 ~4500ms 降至 ~50-100ms
- GPU 利用率从 2.2% 提升至 70-80%

## 📋 前置条件

```bash
pip install huggingface-hub webdataset torch numpy
```

## 🚀 快速开始

### 方案一：使用示例脚本（推荐新手）

```bash
# 编辑脚本中的路径
vim scripts/build_dataset_example.sh

# 运行
bash scripts/build_dataset_example.sh
```

### 方案二：手动运行（推荐进阶用户）

#### 1️⃣ 先处理验证集（测试流程）

```bash
python scripts/build_preprocessed_dataset.py \
    --split valid \
    --output-dir /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/valid_preprocessed_webdataset \
    --samples-per-shard 128 \
    --num-workers 8
```

**预期输出**：
```
📋 Listing files from ibrahimhamamci/CT-RATE (split=valid)...
   Found 7686 valid files
📦 Grouped 7686 files into 60 shards (128 samples/shard)
✅ Found 0/60 existing shards
⚠️  Missing 60 shards
🔄 Processing 60 missing shards...
Processing shards: 100%|████████| 60/60 [15:30<00:00, 15.5s/shard]
📄 Generated manifest: .../manifest.json
   Total samples: 7686
   Total shards: 60
```

#### 2️⃣ 处理训练集

```bash
python scripts/build_preprocessed_dataset.py \
    --split train \
    --output-dir /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/train_preprocessed_webdataset \
    --samples-per-shard 128 \
    --num-workers 16
```

**预期输出**：
```
📋 Listing files from ibrahimhamamci/CT-RATE (split=train)...
   Found 40279 train files
📦 Grouped 40279 files into 315 shards (128 samples/shard)
✅ Found 0/315 existing shards
⚠️  Missing 315 shards
🔄 Processing 315 missing shards...
Processing shards: 100%|████████| 315/315 [82:15<00:00, 15.7s/shard]
```

## 📊 Manifest 文件

每个数据集都会生成 `manifest.json`，记录数据集信息：

```json
{
  "dataset": "CT-RATE",
  "split": "train",
  "format": "webdataset",
  "preprocessed": true,
  "total_shards": 315,
  "total_samples": 40279,
  "sample_shape": [480, 480, 240],
  "sample_dtype": "float16",
  "num_classes": 18,
  "shards": [
    {
      "shard_index": 0,
      "filename": "shard-000000.tar",
      "num_samples": 128,
      "size_bytes": 14155776
    },
    ...
  ]
}
```

## 🔧 高级选项

### 增量处理（续传）

脚本会自动检测已存在的 shards，只处理缺失的部分：

```bash
# 如果中断，直接重新运行即可续传
python scripts/build_preprocessed_dataset.py \
    --split train \
    --output-dir /path/to/output \
    --num-workers 16
```

### 强制重新处理

```bash
python scripts/build_preprocessed_dataset.py \
    --split train \
    --output-dir /path/to/output \
    --force  # 重新处理所有 shards
```

### 自定义 shard 大小

```bash
# 每个 shard 包含 256 个样本（更大的文件，更少的 shards）
python scripts/build_preprocessed_dataset.py \
    --split train \
    --output-dir /path/to/output \
    --samples-per-shard 256
```

### 调整并行度

```bash
# 根据 CPU 核心数调整
python scripts/build_preprocessed_dataset.py \
    --split train \
    --output-dir /path/to/output \
    --num-workers 32  # 更多并行下载和处理
```

## 📁 输出目录结构

```
valid_preprocessed_webdataset/
├── manifest.json           # 数据集元信息
├── shard-000000.tar        # Shard 0 (128 samples)
├── shard-000001.tar        # Shard 1 (128 samples)
├── ...
└── shard-000059.tar        # Shard 59

train_preprocessed_webdataset/
├── manifest.json
├── shard-000000.tar
├── shard-000001.tar
├── ...
└── shard-000314.tar        # Shard 314 (最后一个可能不满 128)
```

每个 tar 文件内部结构（WebDataset 格式）：
```
shard-000000.tar
├── sample_001.bin          # 预处理后的 volume (480x480x240 float16)
├── sample_001.txt          # 报告文本
├── sample_001.cls          # 疾病标签 (18 classes)
├── sample_001.json         # 元数据
├── sample_002.bin
├── sample_002.txt
├── ...
```

## 🔄 更新训练配置

处理完成后，更新你的配置文件：

```yaml
data:
  # 使用预处理后的数据集
  train_shard_pattern: "/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/train_preprocessed_webdataset/shard-{000000..000314}.tar"
  valid_shard_pattern: "/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/valid_preprocessed_webdataset/shard-{000000..000059}.tar"

  # 启用快速加载模式
  preprocessed: true

  # 可以减少 num_workers（预处理已完成，不需要那么多 CPU）
  num_workers: 8  # 从 24 降至 8
```

## ✅ 验证数据正确性

如果你之前有 `train_fixed_webdataset` 数据，可以验证预处理是否正确：

```bash
python scripts/verify_preprocessed_data.py \
    --original-pattern "/path/to/train_fixed_webdataset/shard-{000000..000001}.tar" \
    --preprocessed-pattern "/path/to/train_preprocessed_webdataset/shard-{000000..000001}.tar" \
    --num-samples 10
```

**预期输出**：
```
✅ All samples passed verification!
```

## 💾 存储空间估算

- **原始数据** (npz, 变长)：约 14TB
- **预处理数据** (固定大小)：约 4TB
  - 每个样本：480 × 480 × 240 × 2 bytes = 110 MB
  - 40,279 训练样本：约 4.3 TB
  - 7,686 验证样本：约 822 GB

## ⏱️ 处理时间估算

基于 num_workers=16：

- **验证集**（7,686 samples）：约 15-20 分钟
- **训练集**（40,279 samples）：约 80-120 分钟

实际时间取决于：
- 网络速度（下载 HF 数据）
- CPU 核心数（并行处理）
- 磁盘 I/O 速度

## 🐛 故障排除

### 问题 1：下载失败

```bash
❌ Failed to download dataset/train_fixed/sample_001.npz: Connection timeout
```

**解决方案**：重新运行脚本，它会自动续传，只处理缺失的 shards。

### 问题 2：内存不足

```bash
MemoryError: Unable to allocate array
```

**解决方案**：减少 `--num-workers`：

```bash
python scripts/build_preprocessed_dataset.py \
    --split train \
    --output-dir /path/to/output \
    --num-workers 4  # 降低并行度
```

### 问题 3：磁盘空间不足

**解决方案**：
1. 先处理一部分数据
2. 脚本会自动清理临时下载的文件
3. 确保至少有 5TB 可用空间

### 问题 4：HuggingFace 认证

如果数据集需要认证：

```bash
# 设置 HF token
export HF_TOKEN="your_token_here"

# 或者使用 huggingface-cli
huggingface-cli login
```

## 📈 性能提升

使用预处理数据后的预期提升：

| 指标 | 之前 | 之后 | 提升 |
|------|------|------|------|
| 数据加载时间 | ~4500ms | ~50-100ms | **45-90x** |
| GPU 利用率 | 2.2% | 70-80% | **32-36x** |
| 整体训练速度 | 4.8s/step | ~0.5s/step | **~10x** |
| CPU 核心需求 | 60 threads | 16 threads | **节省 73%** |

## 🔍 工作原理

### 原始流程（慢）
```
训练循环每一步：
1. 从 tar 读取 npz (100ms)
2. 解压 npz (50ms)
3. Rescale (250ms)
4. Clip (127ms)
5. Resize (262ms)
6. Normalize (135ms)
7. Crop/Pad (50ms)
8. GPU 操作 (379ms)
────────────────────────
总计：~1350ms/step
```

### 预处理流程（快）
```
一次性预处理：
1-7. 所有预处理操作 → 保存为 WebDataset

训练循环每一步：
1. 从 tar 读取已处理数据 (30ms)
2. Permute + Unsqueeze (0.02ms)
3. GPU 操作 (379ms)
────────────────────────
总计：~410ms/step
```

## 📚 相关脚本

- `build_preprocessed_dataset.py` - 主脚本（从 HF 构建预处理数据集）
- `preprocess_webdataset.py` - 转换已有的 WebDataset
- `verify_preprocessed_data.py` - 验证预处理正确性
- `inspect_webdataset.py` - 检查 WebDataset 内容

## 💡 提示

1. **先测试小数据集**：先处理验证集（更小），确认流程正确
2. **使用 tmux/screen**：处理训练集需要 1-2 小时，使用持久会话
3. **监控进度**：脚本会显示进度条和成功/失败统计
4. **保留 manifest**：`manifest.json` 包含重要的数据集信息
5. **增量处理**：中断后重新运行会自动续传

## 📞 获取帮助

```bash
python scripts/build_preprocessed_dataset.py --help
```
