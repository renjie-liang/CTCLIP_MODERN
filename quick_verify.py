#!/usr/bin/env python3
"""Quick verification of one sample"""
import json
import numpy as np
import torch
import webdataset as wds

dataset = wds.WebDataset("/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/train_subset_50/shard-000000.tar", shardshuffle=False)

for i, sample in enumerate(dataset):
    if i >= 1:
        break

    print("✅ Sample structure:")
    print(f"   Keys: {[k for k in sample.keys() if not k.startswith('__')]}")

    metadata = json.loads(sample['json'].decode('utf-8'))
    print(f"\n✅ Metadata: {metadata}")

    volume_data = np.frombuffer(sample['bin'], dtype=np.float16).reshape(480, 480, 240).copy()
    print(f"\n✅ Volume shape: {volume_data.shape} (H, W, D)")
    print(f"   Value range: [{volume_data.min():.3f}, {volume_data.max():.3f}]")

    volume_tensor = torch.from_numpy(volume_data).float()
    volume_tensor = volume_tensor.permute(2, 0, 1).unsqueeze(0)
    print(f"\n✅ After DataLoader transform: {volume_tensor.shape} (C, D, H, W)")

    labels = np.frombuffer(sample['labels'], dtype=np.float32)
    print(f"\n✅ Labels: shape={labels.shape}, unique={np.unique(labels)}")

    report = sample['txt'].decode('utf-8')
    print(f"\n✅ Report: {len(report)} chars")
    print(f"   Preview: {report[:100]}...")

print("\n🎉 数据格式验证通过！")
