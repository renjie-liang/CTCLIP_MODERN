#!/usr/bin/env python3
"""Test script to verify split parameter handling"""

import sys
sys.path.insert(0, '/home/user/CTCLIP_MODERN/scripts_important')

from config_npz_conversion import SPLIT_CONFIGS, LOCAL_SOURCE_DIRS

def test_split_config(split='valid'):
    """Test what configuration is used for a given split"""
    print(f"\n{'='*80}")
    print(f"Testing split parameter: '{split}'")
    print(f"{'='*80}\n")

    # Check SPLIT_CONFIGS
    if split in SPLIT_CONFIGS:
        config = SPLIT_CONFIGS[split]
        print(f"✓ SPLIT_CONFIGS['{split}']:")
        print(f"  - hf_path_pattern: '{config['hf_path_pattern']}'")
        print(f"  - output_dir: '{config['output_dir']}'")
    else:
        print(f"✗ '{split}' not found in SPLIT_CONFIGS")

    # Check LOCAL_SOURCE_DIRS
    print()
    if split in LOCAL_SOURCE_DIRS:
        local_dir = LOCAL_SOURCE_DIRS[split]
        print(f"✓ LOCAL_SOURCE_DIRS['{split}']:")
        print(f"  - path: '{local_dir}'")
    else:
        print(f"✗ '{split}' not found in LOCAL_SOURCE_DIRS")

    # Simulate file filtering
    print()
    print("Simulated file filtering:")
    test_files = [
        'dataset/train_fixed/train_001/train_001_a/train_001_a_1.nii.gz',
        'dataset/train_fixed/train_002/train_002_a/train_002_a_1.nii.gz',
        'dataset/valid_fixed/valid_001/valid_001_a/valid_001_a_1.nii.gz',
        'dataset/valid_fixed/valid_002/valid_002_a/valid_002_a_1.nii.gz',
    ]

    path_pattern = config['hf_path_pattern']
    matched_files = [
        f for f in test_files
        if f.startswith(path_pattern) and f.endswith('.nii.gz')
    ]

    print(f"  Pattern: '{path_pattern}'")
    print(f"  Matched {len(matched_files)} files:")
    for f in matched_files:
        print(f"    - {f}")

if __name__ == "__main__":
    # Test both splits
    test_split_config('valid')
    test_split_config('train')
