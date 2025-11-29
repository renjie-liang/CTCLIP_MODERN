#!/bin/bash
# Script to create WebDataset subsets using symlinks

set -e  # Exit on error

# Configuration
BASE_DIR="/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset"
TRAIN_SOURCE="${BASE_DIR}/train_preprocessed_webdataset"
VAL_SOURCE="${BASE_DIR}/valid_preprocessed_webdataset"

# Subset configuration
TRAIN_SHARDS=50  # 0-49
VAL_SHARDS=10    # 0-9

# Create train subset
echo "Creating training subset (${TRAIN_SHARDS} shards)..."
TRAIN_SUBSET="${BASE_DIR}/train_subset_${TRAIN_SHARDS}"
mkdir -p "${TRAIN_SUBSET}"

cd "${TRAIN_SUBSET}"

# Create symlinks for training shards (000000-000049)
echo "  Creating symlinks..."
for i in $(seq -f "%06g" 0 $((TRAIN_SHARDS - 1))); do
    if [ -f "${TRAIN_SOURCE}/shard-${i}.tar" ]; then
        ln -sf "${TRAIN_SOURCE}/shard-${i}.tar" "shard-${i}.tar"
    else
        echo "  Warning: ${TRAIN_SOURCE}/shard-${i}.tar not found"
    fi
done

# Calculate approximate sample count
# Original: 317 shards = 47319 samples
# Formula: (50 / 317) * 47319 ≈ 7462
TRAIN_SAMPLES=$((47319 * TRAIN_SHARDS / 317))

# Create manifest.json for training
echo "  Creating manifest.json..."
cat > manifest.json << EOF
{
  "total_samples": ${TRAIN_SAMPLES},
  "num_shards": ${TRAIN_SHARDS},
  "samples_per_shard_avg": $((TRAIN_SAMPLES / TRAIN_SHARDS)),
  "description": "Subset of training data (shards 0-$((TRAIN_SHARDS - 1)))",
  "created_from": "${TRAIN_SOURCE}"
}
EOF

echo "  ✓ Training subset created: ${TRAIN_SUBSET}"
echo "    Shards: ${TRAIN_SHARDS}"
echo "    Estimated samples: ${TRAIN_SAMPLES}"

# Create validation subset
echo ""
echo "Creating validation subset (${VAL_SHARDS} shards)..."
VAL_SUBSET="${BASE_DIR}/val_subset_${VAL_SHARDS}"
mkdir -p "${VAL_SUBSET}"

cd "${VAL_SUBSET}"

# Create symlinks for validation shards (000000-000009)
echo "  Creating symlinks..."
for i in $(seq -f "%06g" 0 $((VAL_SHARDS - 1))); do
    if [ -f "${VAL_SOURCE}/shard-${i}.tar" ]; then
        ln -sf "${VAL_SOURCE}/shard-${i}.tar" "shard-${i}.tar"
    else
        echo "  Warning: ${VAL_SOURCE}/shard-${i}.tar not found"
    fi
done

# Calculate approximate sample count
# Original: 61 shards = 3039 samples
# Formula: (10 / 61) * 3039 ≈ 498
VAL_SAMPLES=$((3039 * VAL_SHARDS / 61))

# Create manifest.json for validation
echo "  Creating manifest.json..."
cat > manifest.json << EOF
{
  "total_samples": ${VAL_SAMPLES},
  "num_shards": ${VAL_SHARDS},
  "samples_per_shard_avg": $((VAL_SAMPLES / VAL_SHARDS)),
  "description": "Subset of validation data (shards 0-$((VAL_SHARDS - 1)))",
  "created_from": "${VAL_SOURCE}"
}
EOF

echo "  ✓ Validation subset created: ${VAL_SUBSET}"
echo "    Shards: ${VAL_SHARDS}"
echo "    Estimated samples: ${VAL_SAMPLES}"

# Summary
echo ""
echo "=========================================="
echo "Subset Creation Complete!"
echo "=========================================="
echo ""
echo "Training subset:"
echo "  Path: ${TRAIN_SUBSET}"
echo "  Pattern: ${TRAIN_SUBSET}/shard-{000000..000049}.tar"
echo "  Samples: ~${TRAIN_SAMPLES}"
echo ""
echo "Validation subset:"
echo "  Path: ${VAL_SUBSET}"
echo "  Pattern: ${VAL_SUBSET}/shard-{000000..000009}.tar"
echo "  Samples: ~${VAL_SAMPLES}"
echo ""
echo "Update your config with:"
echo "  webdataset_shards_train: ${TRAIN_SUBSET}/shard-{000000..000049}.tar"
echo "  webdataset_shards_val: ${VAL_SUBSET}/shard-{000000..000009}.tar"
echo ""
