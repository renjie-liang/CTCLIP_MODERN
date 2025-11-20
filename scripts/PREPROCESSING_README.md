# CT-RATE Data Preprocessing Guide

This guide explains how to build preprocessed WebDataset directly from Hugging Face to achieve **10x training acceleration**.

## 🎯 Goals

- Complete CPU-intensive preprocessing (resize, normalize, etc.) ahead of time
- Only need to quickly load preprocessed data during training
- Reduce data loading from ~4500ms to ~50-100ms
- Increase GPU utilization from 2.2% to 70-80%

## 📋 Prerequisites

```bash
pip install huggingface-hub webdataset torch numpy
```

## 🚀 Quick Start

### Solution 1: Using Example Script (Recommended for Beginners)

```bash
# Edit paths in the script
vim scripts/build_dataset_example.sh

# Run
bash scripts/build_dataset_example.sh
```

### Solution 2: Manual Execution (Recommended for Advanced Users)

#### 1️⃣ Process Validation Set First (Test Workflow)

```bash
python scripts/build_preprocessed_dataset.py \
    --split valid \
    --output-dir /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/valid_preprocessed_webdataset \
    --samples-per-shard 128 \
    --num-workers 8
```

**Expected Output**:
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

#### 2️⃣ Process Training Set

```bash
python scripts/build_preprocessed_dataset.py \
    --split train \
    --output-dir /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/train_preprocessed_webdataset \
    --samples-per-shard 128 \
    --num-workers 16
```

**Expected Output**:
```
📋 Listing files from ibrahimhamamci/CT-RATE (split=train)...
   Found 40279 train files
📦 Grouped 40279 files into 315 shards (128 samples/shard)
✅ Found 0/315 existing shards
⚠️  Missing 315 shards
🔄 Processing 315 missing shards...
Processing shards: 100%|████████| 315/315 [82:15<00:00, 15.7s/shard]
```

## 📊 Manifest File

Each dataset will generate a `manifest.json` that records dataset information:

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

## 🔧 Advanced Options

### Incremental Processing (Resume)

The script automatically detects existing shards and only processes missing ones:

```bash
# If interrupted, simply re-run to resume
python scripts/build_preprocessed_dataset.py \
    --split train \
    --output-dir /path/to/output \
    --num-workers 16
```

### Force Reprocessing

```bash
python scripts/build_preprocessed_dataset.py \
    --split train \
    --output-dir /path/to/output \
    --force  # Reprocess all shards
```

### Customize Shard Size

```bash
# Each shard contains 256 samples (larger files, fewer shards)
python scripts/build_preprocessed_dataset.py \
    --split train \
    --output-dir /path/to/output \
    --samples-per-shard 256
```

### Adjust Parallelism

```bash
# Adjust based on CPU core count
python scripts/build_preprocessed_dataset.py \
    --split train \
    --output-dir /path/to/output \
    --num-workers 32  # More parallel downloading and processing
```

## 📁 Output Directory Structure

```
valid_preprocessed_webdataset/
├── manifest.json           # Dataset metadata
├── shard-000000.tar        # Shard 0 (128 samples)
├── shard-000001.tar        # Shard 1 (128 samples)
├── ...
└── shard-000059.tar        # Shard 59

train_preprocessed_webdataset/
├── manifest.json
├── shard-000000.tar
├── shard-000001.tar
├── ...
└── shard-000314.tar        # Shard 314 (last one may have less than 128)
```

Internal structure of each tar file (WebDataset format):
```
shard-000000.tar
├── sample_001.bin          # Preprocessed volume (480x480x240 float16)
├── sample_001.txt          # Report text
├── sample_001.cls          # Disease labels (18 classes)
├── sample_001.json         # Metadata
├── sample_002.bin
├── sample_002.txt
├── ...
```

## 🔄 Update Training Config

After processing is complete, update your config file:

```yaml
data:
  # Use preprocessed dataset
  train_shard_pattern: "/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/train_preprocessed_webdataset/shard-{000000..000314}.tar"
  valid_shard_pattern: "/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/valid_preprocessed_webdataset/shard-{000000..000059}.tar"

  # Enable fast loading mode
  preprocessed: true

  # Can reduce num_workers (preprocessing complete, don't need as many CPUs)
  num_workers: 8  # Reduce from 24 to 8
```

## ✅ Verify Data Correctness

If you previously had `train_fixed_webdataset` data, you can verify preprocessing correctness:

```bash
python scripts/verify_preprocessed_data.py \
    --original-pattern "/path/to/train_fixed_webdataset/shard-{000000..000001}.tar" \
    --preprocessed-pattern "/path/to/train_preprocessed_webdataset/shard-{000000..000001}.tar" \
    --num-samples 10
```

**Expected Output**:
```
✅ All samples passed verification!
```

## 💾 Storage Space Estimation

- **Original data** (npz, variable length): ~14TB
- **Preprocessed data** (fixed size): ~4TB
  - Per sample: 480 × 480 × 240 × 2 bytes = 110 MB
  - 40,279 training samples: ~4.3 TB
  - 7,686 validation samples: ~822 GB

## ⏱️ Processing Time Estimation

Based on num_workers=16:

- **Validation set** (7,686 samples): ~15-20 minutes
- **Training set** (40,279 samples): ~80-120 minutes

Actual time depends on:
- Network speed (downloading HF data)
- CPU core count (parallel processing)
- Disk I/O speed

## 🐛 Troubleshooting

### Issue 1: Download Failed

```bash
❌ Failed to download dataset/train_fixed/sample_001.npz: Connection timeout
```

**Solution**: Re-run the script, it will automatically resume and only process missing shards.

### Issue 2: Out of Memory

```bash
MemoryError: Unable to allocate array
```

**Solution**: Reduce `--num-workers`:

```bash
python scripts/build_preprocessed_dataset.py \
    --split train \
    --output-dir /path/to/output \
    --num-workers 4  # Lower parallelism
```

### Issue 3: Insufficient Disk Space

**Solution**:
1. Process data in parts first
2. Script automatically cleans up temporary downloaded files
3. Ensure at least 5TB available space

### Issue 4: HuggingFace Authentication

If dataset requires authentication:

```bash
# Set HF token
export HF_TOKEN="your_token_here"

# Or use huggingface-cli
huggingface-cli login
```

## 📈 Performance Improvement

Expected improvements after using preprocessed data:

| Metric | Before | After | Improvement |
|------|------|------|------|
| Data loading time | ~4500ms | ~50-100ms | **45-90x** |
| GPU utilization | 2.2% | 70-80% | **32-36x** |
| Overall training speed | 4.8s/step | ~0.5s/step | **~10x** |
| CPU core requirement | 60 threads | 16 threads | **Save 73%** |

## 🔍 How It Works

### Original Workflow (Slow)
```
Training loop each step:
1. Read npz from tar (100ms)
2. Decompress npz (50ms)
3. Rescale (250ms)
4. Clip (127ms)
5. Resize (262ms)
6. Normalize (135ms)
7. Crop/Pad (50ms)
8. GPU operations (379ms)
────────────────────────
Total: ~1350ms/step
```

### Preprocessing Workflow (Fast)
```
One-time preprocessing:
1-7. All preprocessing operations → Save as WebDataset

Training loop each step:
1. Read preprocessed data from tar (30ms)
2. Permute + Unsqueeze (0.02ms)
3. GPU operations (379ms)
────────────────────────
Total: ~410ms/step
```

## 📚 Related Scripts

- `build_preprocessed_dataset.py` - Main script (build preprocessed dataset from HF)
- `preprocess_webdataset.py` - Convert existing WebDataset
- `verify_preprocessed_data.py` - Verify preprocessing correctness
- `inspect_webdataset.py` - Inspect WebDataset contents

## 💡 Tips

1. **Test with small dataset first**: Process validation set (smaller) first to confirm workflow is correct
2. **Use tmux/screen**: Processing training set takes 1-2 hours, use persistent session
3. **Monitor progress**: Script shows progress bar and success/failure statistics
4. **Keep manifest**: `manifest.json` contains important dataset information
5. **Incremental processing**: Re-running after interruption will automatically resume

## 📞 Get Help

```bash
python scripts/build_preprocessed_dataset.py --help
```
