#!/bin/bash
# Example script to build preprocessed dataset from Hugging Face
# This script demonstrates how to use build_preprocessed_dataset.py (Simplified version)

set -e  # Exit on error

# Configuration
REPO_ID="ibrahimhamamci/CT-RATE"
BASE_DIR="/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset"
TEMP_DIR="/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/temp"  # Custom temp directory (instead of /tmp)

echo "=========================================="
echo "Build Preprocessed CT-RATE Dataset"
echo "=========================================="
echo "Repo: $REPO_ID"
echo "Base dir: $BASE_DIR"
echo ""

# Step 1: Build validation set (smaller, faster to test)
echo "Step 1: Building validation set..."
python scripts/build_preprocessed_dataset.py \
    --split valid \
    --output-dir "$BASE_DIR/valid_preprocessed_webdataset" \
    --repo-id "$REPO_ID" \
    --temp-dir "$TEMP_DIR"

echo ""
echo "✅ Validation set completed!"
echo ""

# Step 2: Build training set
echo "Step 2: Building training set..."
python scripts/build_preprocessed_dataset.py \
    --split train \
    --output-dir "$BASE_DIR/train_preprocessed_webdataset" \
    --repo-id "$REPO_ID" \
    --temp-dir "$TEMP_DIR"

echo ""
echo "✅ Training set completed!"
echo ""

# Step 3: Verify manifests
echo "Step 3: Checking manifests..."
echo ""
echo "Validation manifest:"
cat "$BASE_DIR/valid_preprocessed_webdataset/manifest.json" | head -20
echo ""
echo "Training manifest:"
cat "$BASE_DIR/train_preprocessed_webdataset/manifest.json" | head -20

echo ""
echo "=========================================="
echo "🎉 All done!"
echo "=========================================="
echo "Features:"
echo "- File-level resume: only processes missing files"
echo "- Appends to incomplete shards automatically"
echo "- Single-threaded, simple, easy to debug"
echo "- Auto-infers SAMPLES_PER_SHARD from existing shards"
echo ""
echo "Next steps:"
echo "1. Update your config to use the preprocessed datasets"
echo "2. Set preprocessed: true in the config"
echo "3. Train with 10x faster data loading!"
echo ""
