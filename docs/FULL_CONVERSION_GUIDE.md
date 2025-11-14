# 全量转换NPZ到WebDataset - 删除源文件模式

## ⚠️ 重要警告

**使用 `--delete_source_files` 会永久删除原始NPZ文件！**

- ✅ 只有**成功转换**的文件会被删除
- ⚠️ 删除操作**不可撤销**
- 📊 预计释放空间：~11-12 TB（对于47K样本）

---

## 推荐的分步骤转换流程

### 步骤 1：小规模测试（100个样本）

先测试100个样本，验证转换正确性：

```bash
python scripts/convert_npz_to_webdataset.py \
  --data_folder /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/train_fixed_npz \
  --reports_file /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/radiology_text_reports/train_reports.csv \
  --meta_file /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/metadata/train_metadata.csv \
  --labels_file /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/multi_abnormality_labels/train_predicted_labels.csv \
  --output_dir /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/train_fixed_webdataset \
  --samples_per_shard 100 \
  --num_workers 16 \
  --test_mode \
  --delete_source_files  # 测试删除功能
```

**验证**：
```bash
# 检查转换结果
python scripts/test_webdataset.py \
  --webdataset_dir /path/to/webdataset/train \
  --num_samples 10 \
  --check_precision

# 确认NPZ文件已被删除（应该少了100个）
find /orange/.../train_fixed_npz -name "*.npz" | wc -l
```

### 步骤 2：分批全量转换（推荐）

为了安全，分批转换，每批5000个样本：

```bash
# 第一批：样本 0-4999
python scripts/convert_npz_to_webdataset.py \
  --data_folder /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/train_fixed_npz \
  --reports_file /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/radiology_text_reports/train_reports.csv \
  --meta_file /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/metadata/train_metadata.csv \
  --labels_file /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/multi_abnormality_labels/train_predicted_labels.csv \
  --output_dir /path/to/webdataset/train_batch1 \
  --samples_per_shard 150 \
  --num_workers 16 \
  --delete_source_files \
  --yes  

# 等第一批成功后，继续下一批...
```

**优点**：
- 如果出错可以及时发现
- 空间逐步释放
- 可以随时中断

### 步骤 3：一次性全量转换（高级）

**仅在确认测试成功后使用！**

```bash
python scripts/convert_npz_to_webdataset.py \
  --data_folder /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/train_fixed_npz \
  --reports_file /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/radiology_text_reports/train_reports.csv \
  --meta_file /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/metadata/train_metadata.csv \
  --labels_file /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/multi_abnormality_labels/train_predicted_labels.csv \
  --output_dir /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/train_fixed_webdataset \
  --samples_per_shard 150 \
  --num_workers 16 \
  --delete_source_files \
  --yes  


```

python scripts/convert_npz_to_webdataset.py \
  --data_folder /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/valid_fixed_npz \
  --reports_file /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/radiology_text_reports/validation_reports.csv \
  --meta_file /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/metadata/validation_metadata.csv \
  --labels_file /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/multi_abnormality_labels/valid_predicted_labels.csv \
  --output_dir /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/valid_fixed_webdataset \
  --samples_per_shard 50 \
  --num_workers 8 \
  --yes


**交互式确认**：
```
⚠️  WARNING: SOURCE FILE DELETION ENABLED ⚠️
This will DELETE 47149 NPZ files after conversion!
Total estimated size: ~16500.2 GB will be freed

IMPORTANT:
  - Files will be deleted IMMEDIATELY after successful conversion
  - This operation CANNOT be undone
  - Make sure you have backups if needed

Are you ABSOLUTELY sure you want to proceed? (type 'DELETE' to confirm): DELETE

✓ Confirmed. Source files will be deleted after conversion.
```

---

## 预期时间和空间

### 时间估算（47K训练集样本）

| Workers | 每个样本时间 | 总时间估算 |
|---------|------------|-----------|
| 8 | ~5秒 | ~65小时 |
| 16 | ~3秒 | ~39小时 |
| 32 | ~2秒 | ~26小时 |

**建议**：使用 16 workers，预留 **40-50小时**

### 空间变化

```
初始状态：
  - NPZ文件：14 TB

转换过程中（最坏情况）：
  - NPZ文件：14 TB（未删除）
  - WebDataset：2.4 TB（部分生成）
  - 峰值使用：16.4 TB

转换完成后：
  - NPZ文件：0 TB（已删除）
  - WebDataset：2.4 TB
  - 最终使用：2.4 TB
  - 释放空间：11.6 TB ✓
```

**关键**：删除是**边转换边删除**，所以峰值空间不会是14+2.4=16.4TB，而是逐步减少！

实际峰值空间取决于：
- 每个shard的样本数（100个）
- 并行workers数（16个）
- 最坏峰值 ≈ 当前NPZ + (16 workers × 100 samples/shard × 230 MB/sample)
- 最坏峰值 ≈ 当前NPZ + 370 GB

---

## 监控进度

### 实时查看转换进度

```bash
# 终端1：运行转换
python scripts/convert_npz_to_webdataset.py ...

# 终端2：监控空间释放
watch -n 10 'df -h /orange/... && echo && find .../train_fixed_npz -name "*.npz" | wc -l'
```

### 查看已转换的样本数

```bash
# 查看manifest
cat /path/to/webdataset/train/manifest.json

# 统计剩余NPZ文件
find /orange/.../train_fixed_npz -name "*.npz" | wc -l
```

---

## 安全检查清单

在运行全量转换前，确认：

- [ ] 已成功运行测试模式（100个样本）
- [ ] 已验证WebDataset数据正确性（test_webdataset.py）
- [ ] 已确认float16精度可接受
- [ ] 已检查输出目录有足够空间（至少3TB）
- [ ] 已了解删除是不可逆的
- [ ] （可选）已备份关键样本
- [ ] 已准备好监控脚本
- [ ] 已预留足够时间（40-50小时）

---

## 常见问题

### Q1: 转换中断了怎么办？

**A**: 重新运行相同命令，脚本会：
- 跳过已转换的样本（通过manifest检查）
- 继续转换剩余样本
- 已删除的NPZ文件不会影响

### Q2: 部分文件转换失败会怎样？

**A**:
- ✓ 失败的文件**不会**被删除
- ✓ 成功的文件继续转换和删除
- ⚠️ 最后会显示警告：`X files were NOT deleted`

### Q3: 可以中途取消吗？

**A**:
- ✓ 可以Ctrl+C中断
- ⚠️ 但已转换的NPZ文件可能已被删除
- 建议：先小批量测试

### Q4: 验证集也要转换吗？

**A**:
- 建议：**训练集转换 + 删除**（频繁使用）
- 建议：**验证集只转换，不删除**（使用少，保险）

```bash
# 验证集：不删除源文件
python scripts/convert_npz_to_webdataset.py \
  --data_folder /orange/.../valid_fixed_npz \
  ... \
  --output_dir /path/to/webdataset/val
  # 注意：没有 --delete_source_files
```

### Q5: 如何估算我的实际压缩率？

**A**: 运行测试模式后查看：
```bash
cat /path/to/webdataset/train/manifest.json
```

```json
{
  "average_sample_size_mb": 230.86,  // 实际压缩后大小
  ...
}
```

压缩率 = 350 MB (原始NPZ) / 230.86 MB ≈ **1.52x**

---

## 最佳实践总结

### 推荐做法 ✅

1. **先测试**：用 `--test_mode` 转换100个样本
2. **验证数据**：用 `test_webdataset.py` 检查
3. **分批转换**：每批5000个，逐步释放空间
4. **监控进度**：用watch命令实时查看
5. **保留验证集**：验证集NPZ不删除作为保险

### 危险做法 ⚠️

1. ❌ 未测试就全量转换+删除
2. ❌ 使用 `--yes` 跳过确认（除非你100%确定）
3. ❌ 转换到同一个目录（会覆盖）
4. ❌ 空间不足时强行转换
5. ❌ 转换过程中手动删除文件

---

## 开始全量转换

确认清单完成后，执行：

```bash
# 训练集全量转换（带删除）
python scripts/convert_npz_to_webdataset.py \
  --data_folder /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/train_fixed_npz \
  --reports_file /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/radiology_text_reports/train_reports.csv \
  --meta_file /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/metadata/train_metadata.csv \
  --labels_file /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/multi_abnormality_labels/train_predicted_labels.csv \
  --output_dir /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/webdataset/train \
  --samples_per_shard 100 \
  --num_workers 16 \
  --delete_source_files
```

**预期结果**：
- 转换时间：~40小时
- 释放空间：~11.6 TB
- WebDataset大小：~2.4 TB
- 压缩率：~5.8x

祝转换顺利！🚀
