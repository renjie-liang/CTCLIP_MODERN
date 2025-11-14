#!/usr/bin/env python3
"""
Inspect WebDataset to understand current data format
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import webdataset as wds
import numpy as np
import json


def inspect_samples(shard_pattern, num_samples=5):
    """Inspect first N samples from WebDataset"""

    dataset = wds.WebDataset(shard_pattern, shardshuffle=False)

    print("="*80)
    print("Inspecting WebDataset Format")
    print("="*80)
    print(f"Pattern: {shard_pattern}\n")

    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break

        # Decode metadata
        metadata = json.loads(sample['json'].decode('utf-8'))

        # Decode volume
        volume_shape = tuple(metadata['volume_shape'])
        volume_dtype = np.dtype(metadata['volume_dtype'])
        volume_data = np.frombuffer(sample['bin'], dtype=volume_dtype).reshape(volume_shape)

        # Decode labels
        labels = np.frombuffer(sample['labels'], dtype=np.float32)

        print(f"Sample {i+1}:")
        print(f"  Study ID: {metadata['study_id']}")
        print(f"  Volume shape: {volume_shape}")
        print(f"  Volume dtype: {volume_dtype}")
        print(f"  Volume size: {volume_data.nbytes / 1024**2:.2f} MB")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Spacing: XY={metadata.get('XYSpacing')}, Z={metadata.get('ZSpacing')}")
        print(f"  RescaleSlope: {metadata.get('RescaleSlope')}")
        print(f"  RescaleIntercept: {metadata.get('RescaleIntercept')}")
        print()

    print("="*80)


if __name__ == "__main__":
    # Inspect training set
    train_pattern = "/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/train_fixed_webdataset/shard-{000000..000314}.tar"

    print("\n🔍 Inspecting TRAINING set:")
    inspect_samples(train_pattern, num_samples=3)

    # Inspect validation set
    val_pattern = "/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/valid_fixed_webdataset/shard-{000000..000059}.tar"

    print("\n🔍 Inspecting VALIDATION set:")
    inspect_samples(val_pattern, num_samples=3)
