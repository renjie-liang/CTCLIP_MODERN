#!/usr/bin/env python3
"""
Verify WebDataset format and content.
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch
import webdataset as wds

def verify_dataset(shard_pattern: str, num_samples: int = 3):
    """Verify WebDataset format matches expected structure."""

    print("="*80)
    print("WebDataset Verification")
    print("="*80)
    print(f"Pattern: {shard_pattern}\n")

    # Create dataset
    dataset = wds.WebDataset(shard_pattern, shardshuffle=False)

    issues = []

    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break

        print(f"\n{'='*80}")
        print(f"Sample {i+1}")
        print(f"{'='*80}")

        # Check keys
        print(f"\n1️⃣ Keys in sample: {list(sample.keys())}")
        expected_keys = {'__key__', 'bin', 'json', 'txt', 'labels'}
        missing_keys = expected_keys - set(sample.keys())
        extra_keys = set(sample.keys()) - expected_keys

        if missing_keys:
            issue = f"Sample {i+1}: Missing keys: {missing_keys}"
            print(f"   ❌ {issue}")
            issues.append(issue)
        if extra_keys:
            print(f"   ℹ️  Extra keys (OK): {extra_keys}")

        # Decode metadata
        try:
            metadata = json.loads(sample['json'].decode('utf-8'))
            print(f"\n2️⃣ Metadata (json):")
            print(f"   study_id: {metadata.get('study_id')}")
            print(f"   volume_shape: {metadata.get('volume_shape')}")
            print(f"   volume_dtype: {metadata.get('volume_dtype')}")

            # Verify metadata fields
            if 'volume_shape' not in metadata:
                issue = f"Sample {i+1}: Missing 'volume_shape' in metadata"
                print(f"   ❌ {issue}")
                issues.append(issue)
            elif metadata['volume_shape'] != [480, 480, 240]:
                issue = f"Sample {i+1}: Incorrect volume_shape {metadata['volume_shape']}, expected [480, 480, 240]"
                print(f"   ❌ {issue}")
                issues.append(issue)
            else:
                print(f"   ✅ Shape is correct: (H, W, D) = {metadata['volume_shape']}")

            if metadata.get('volume_dtype') != 'float16':
                issue = f"Sample {i+1}: Unexpected volume_dtype {metadata.get('volume_dtype')}, expected float16"
                print(f"   ❌ {issue}")
                issues.append(issue)
            else:
                print(f"   ✅ Dtype is correct: float16")

        except Exception as e:
            issue = f"Sample {i+1}: Failed to decode metadata: {e}"
            print(f"   ❌ {issue}")
            issues.append(issue)
            continue

        # Decode volume
        try:
            print(f"\n3️⃣ Volume data (bin):")
            volume_shape = tuple(metadata['volume_shape'])
            volume_data = np.frombuffer(sample['bin'], dtype=np.float16)
            print(f"   Raw bytes size: {len(sample['bin'])} bytes")
            print(f"   Expected size: {np.prod(volume_shape) * 2} bytes (float16 = 2 bytes)")

            volume_data = volume_data.reshape(volume_shape)
            print(f"   Reshaped to: {volume_data.shape} (H, W, D)")
            print(f"   Value range: [{volume_data.min():.3f}, {volume_data.max():.3f}]")
            print(f"   Mean: {volume_data.mean():.3f}, Std: {volume_data.std():.3f}")

            # Check value range (should be in [-1, 1] after preprocessing)
            if volume_data.min() < -2 or volume_data.max() > 2:
                issue = f"Sample {i+1}: Volume values out of expected range [-1, 1]: [{volume_data.min():.3f}, {volume_data.max():.3f}]"
                print(f"   ⚠️  {issue}")
                issues.append(issue)
            else:
                print(f"   ✅ Value range is reasonable for normalized data")

            # Simulate DataLoader transformation
            print(f"\n4️⃣ DataLoader transformation:")
            volume_tensor = torch.from_numpy(volume_data).float()  # (480, 480, 240)
            print(f"   After torch.from_numpy: {volume_tensor.shape}")

            volume_tensor = volume_tensor.permute(2, 0, 1)  # (H, W, D) -> (D, H, W)
            print(f"   After permute(2,0,1): {volume_tensor.shape} (D, H, W)")

            volume_tensor = volume_tensor.unsqueeze(0)  # Add channel dim
            print(f"   After unsqueeze(0): {volume_tensor.shape} (C, D, H, W)")

            if volume_tensor.shape != (1, 240, 480, 480):
                issue = f"Sample {i+1}: Final tensor shape {volume_tensor.shape} != expected (1, 240, 480, 480)"
                print(f"   ❌ {issue}")
                issues.append(issue)
            else:
                print(f"   ✅ Final tensor shape is correct: (C, D, H, W) = {tuple(volume_tensor.shape)}")

        except Exception as e:
            issue = f"Sample {i+1}: Failed to decode volume: {e}"
            print(f"   ❌ {issue}")
            issues.append(issue)
            continue

        # Decode labels
        try:
            print(f"\n5️⃣ Labels:")
            labels = np.frombuffer(sample['labels'], dtype=np.float32)
            print(f"   Shape: {labels.shape}")
            print(f"   Values: {labels}")
            print(f"   Unique values: {np.unique(labels)}")

            # Check label count (should be 18 for CT-RATE dataset)
            expected_num_classes = 18
            if len(labels) != expected_num_classes:
                issue = f"Sample {i+1}: Label count {len(labels)} != expected {expected_num_classes}"
                print(f"   ❌ {issue}")
                issues.append(issue)
            else:
                print(f"   ✅ Label count is correct: {len(labels)} classes")

        except Exception as e:
            issue = f"Sample {i+1}: Failed to decode labels: {e}"
            print(f"   ❌ {issue}")
            issues.append(issue)

        # Decode text
        try:
            print(f"\n6️⃣ Report text (txt):")
            report_text = sample['txt'].decode('utf-8')
            print(f"   Length: {len(report_text)} characters")
            print(f"   Preview: {report_text[:150]}...")

            if len(report_text) == 0:
                print(f"   ⚠️  Warning: Empty report text")
            else:
                print(f"   ✅ Report text exists")

        except Exception as e:
            issue = f"Sample {i+1}: Failed to decode text: {e}"
            print(f"   ❌ {issue}")
            issues.append(issue)

    # Summary
    print(f"\n{'='*80}")
    print("Verification Summary")
    print(f"{'='*80}")

    if len(issues) == 0:
        print("✅ All checks passed! Dataset format is correct.")
        return 0
    else:
        print(f"❌ Found {len(issues)} issues:")
        for issue in issues:
            print(f"   - {issue}")
        return 1


if __name__ == "__main__":
    # Verify train subset
    train_pattern = "/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/train_subset_50/shard-{000000..000049}.tar"

    print("\n🔍 Verifying TRAIN SUBSET (50 shards):\n")
    exit_code = verify_dataset(train_pattern, num_samples=3)

    sys.exit(exit_code)
