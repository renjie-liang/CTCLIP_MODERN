# NPZ DataLoader 使用指南

本文档说明如何使用新创建的 NPZ DataLoader 以及如何对比 NPZ 和 WebDataset 的加载性能。

## 📁 文件结构

```
src/data/
├── npz_loader.py           # NPZ DataLoader 实现
└── webdataset_loader.py    # WebDataset DataLoader 实现

scripts/
├── test_npz_loader.py                    # NPZ loader 基础功能测试
├── quick_compare_npz_vs_webdataset.py    # 快速性能对比脚本
├── benchmark_npz_vs_webdataset.py        # 完整性能对比脚本
└── verify_npz_files.py                   # 数据文件验证脚本
```

## 🚀 快速开始

### 1. 验证数据文件

首先验证 NPZ 文件和 CSV 文件是否存在：

```bash
python scripts/verify_npz_files.py
```

这将检查：
- NPZ 文件目录是否存在
- NPZ 文件格式是否正确（包含 'volume' 键）
- CSV 文件（reports, metadata, labels）是否存在

### 2. 测试 NPZ Loader 基本功能

```bash
python scripts/test_npz_loader.py
```

这将运行三个测试：
- **测试 1**: 单样本加载（验证形状、数据类型、数值范围）
- **测试 2**: DataLoader 批量加载
- **测试 3**: 加载速度测试（10 个样本）

### 3. 快速性能对比

快速对比 NPZ 和 WebDataset 的加载速度：

```bash
python scripts/quick_compare_npz_vs_webdataset.py --num_samples 50
```

参数说明：
- `--num_samples`: 测试样本数量（默认 50）
- `--batch_size`: 批量大小（默认 1）
- `--num_workers`: DataLoader 工作进程数（默认 0）
- `--npz_dir`: NPZ 文件目录
- `--webdataset_dir`: WebDataset 文件目录

### 4. 完整性能 Benchmark

运行完整的性能对比测试：

```bash
python scripts/benchmark_npz_vs_webdataset.py \
    --npz_dir /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/vaild_npz \
    --webdataset_dir /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/webdataset_val \
    --reports_file /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/radiology_text_reports/validation_reports.csv \
    --meta_file /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/metadata/validation_metadata.csv \
    --labels_file /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/multi_abnormality_labels/valid_predicted_labels.csv \
    --num_samples 100
```

这将测试：
- **NPZ 原始 I/O**: 仅 np.load() 时间
- **NPZ 完整加载**: I/O + 处理（窗位、归一化、转换）
- **WebDataset 完整加载**: I/O + 处理
- **性能对比**: 加速比、吞吐量等

## 📊 NPZ Loader 特性

### 数据预处理

NPZ 文件已经包含预处理后的数据：
- ✅ 统一体素间距（0.75mm × 0.75mm × 1.5mm）
- ✅ 裁剪/填充到固定尺寸（480 × 480 × 240）
- ✅ 存储为 int16 格式（节省空间）

### Loader 处理流程

在加载时，NPZ Loader 只需要：
1. **加载数据**: `np.load(file)['volume']`
2. **应用窗位**: `np.clip(volume, -1000, 1000)`  # 肺窗
3. **归一化**: 归一化到 `[-1, 1]` 范围
4. **转换维度**: `(H, W, D) → (D, H, W)`
5. **添加通道**: `(D, H, W) → (1, D, H, W)`

### 配置参数

```python
from src.data import CTReportNPZDataset

dataset = CTReportNPZDataset(
    data_folder="/path/to/npz/files",
    reports_file="/path/to/reports.csv",
    meta_file="/path/to/metadata.csv",
    labels_file="/path/to/labels.csv",
    min_hu=-1000,    # 窗位最小值（肺窗）
    max_hu=1000,     # 窗位最大值（肺窗）
    mode="val"       # "train" 或 "val"
)
```

窗位选项：
- **肺窗**: `min_hu=-1000, max_hu=1000`
- **软组织窗**: `min_hu=-150, max_hu=250`
- **骨窗**: `min_hu=-500, max_hu=1500`

## 🔍 文件过滤

NPZ Loader 在初始化时自动过滤：
- ✅ 不存在的文件
- ✅ 缺少报告的样本
- ✅ 缺少标签的样本

这确保了数据集的完整性和一致性。

## 📈 性能预期

基于设计，预期性能：

| 指标 | NPZ (原始 I/O) | NPZ (完整) | WebDataset (完整) |
|------|----------------|-----------|------------------|
| 加载时间 | ~50ms | ~100ms | ~50-100ms |
| 吞吐量 | ~20 samples/s | ~10 samples/s | ~10-20 samples/s |

**影响因素**：
- 磁盘 I/O 速度
- CPU 性能（窗位、归一化）
- DataLoader workers 数量
- 批量大小

## 🎯 使用场景

### NPZ Loader 适用于：
- ✅ 需要灵活访问单个样本
- ✅ 随机采样和数据增强
- ✅ 调试和开发阶段
- ✅ 小规模数据集

### WebDataset Loader 适用于：
- ✅ 大规模训练（顺序读取更快）
- ✅ 分布式训练
- ✅ 需要更小的存储空间（float16 压缩）
- ✅ 生产环境

## 🔧 代码示例

### 在训练中使用 NPZ Loader

```python
from torch.utils.data import DataLoader
from src.data import CTReportNPZDataset

# 创建数据集
train_dataset = CTReportNPZDataset(
    data_folder="/path/to/train_npz",
    reports_file="/path/to/train_reports.csv",
    meta_file="/path/to/train_metadata.csv",
    labels_file="/path/to/train_labels.csv",
    min_hu=-1000,
    max_hu=1000,
    mode="train"
)

# 创建 DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)

# 训练循环
for epoch in range(num_epochs):
    for volumes, reports, labels, study_ids in train_loader:
        # volumes: (batch, 1, 240, 480, 480) tensor
        # reports: list of strings
        # labels: (batch, num_classes) array
        # study_ids: list of strings

        # 你的训练代码...
        pass
```

### 切换到 WebDataset

```python
from src.data import CTReportWebDataset

# 创建数据集
train_dataset = CTReportWebDataset(
    shard_pattern="/path/to/shards/shard-{000000..000099}.tar",
    shuffle=True,
    mode="train"
)

# 创建 DataLoader
train_loader = train_dataset.create_pytorch_dataloader(
    batch_size=4,
    num_workers=8
)

# 使用方式相同
for volumes, reports, labels, study_ids, embeddings in train_loader:
    # 你的训练代码...
    pass
```

## 🐛 故障排查

### 问题 1: "No NPZ files found"
**解决方案**: 检查 `data_folder` 路径是否正确

### 问题 2: "Skipped X samples without reports/labels"
**解决方案**: 检查 CSV 文件中的 `VolumeName` 列是否与 NPZ 文件名匹配

### 问题 3: "Wrong shape" 错误
**解决方案**: 检查 NPZ 文件是否包含正确形状的 'volume' 键 (480, 480, 240)

### 问题 4: 加载速度慢
**解决方案**:
- 增加 `num_workers` 参数
- 检查磁盘 I/O 性能
- 考虑使用 SSD 存储 NPZ 文件

## 📝 数据格式要求

### NPZ 文件格式
```python
{
    'volume': np.ndarray,  # shape: (480, 480, 240), dtype: int16
}
```

### CSV 文件格式

**Reports CSV**:
```
VolumeName,Findings_EN,Impressions_EN
train_10670_a_2.nii.gz,"Findings text...","Impressions text..."
```

**Labels CSV**:
```
VolumeName,Atelectasis,Cardiomegaly,Consolidation,...
train_10670_a_2.nii.gz,0,1,0,...
```

**Metadata CSV**:
```
VolumeName,RescaleSlope,RescaleIntercept,XYSpacing,ZSpacing
train_10670_a_2.nii.gz,1.0,0.0,[0.75, 0.75],1.5
```

## 📚 参考资料

- NPZ Loader 实现: `src/data/npz_loader.py`
- WebDataset Loader 实现: `src/data/webdataset_loader.py`
- Benchmark 脚本: `scripts/benchmark_npz_vs_webdataset.py`
