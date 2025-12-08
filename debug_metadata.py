#!/usr/bin/env python3
"""Debug metadata structure"""

import json
import webdataset as wds

# Read first sample
dataset = wds.WebDataset("/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/train_subset_50/shard-000000.tar", shardshuffle=False)

for i, sample in enumerate(dataset):
    if i >= 1:
        break

    print("Sample keys:", list(sample.keys()))
    print("\nRaw JSON bytes:")
    print(sample['json'])
    print("\nDecoded JSON:")
    metadata = json.loads(sample['json'].decode('utf-8'))
    print(json.dumps(metadata, indent=2))
    print("\nAll metadata keys:", list(metadata.keys()))
