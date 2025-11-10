#!/usr/bin/env python3
"""
Quick test script for debugging DataLoader issues.

Usage:
    # Test with timing logs
    DEBUG_TIMING=true python scripts/quick_test_loader.py

    # Test without timing logs
    python scripts/quick_test_loader.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
from src.data.webdataset_loader import CTReportWebDataset
from src.utils.config import load_config


def main():
    print("="*80)
    print("Quick DataLoader Test")
    print("="*80)

    # Load config
    config_path = project_root / "configs" / "base_config.yaml"
    config = load_config(str(config_path))

    # Check if DEBUG_TIMING is enabled
    debug_timing = os.environ.get('DEBUG_TIMING', 'false').lower() == 'true'
    print(f"\nDEBUG_TIMING: {debug_timing}")
    if debug_timing:
        print("Detailed timing information will be printed for each sample\n")

    # Create dataset
    print(f"\nCreating dataset...")
    dataset = CTReportWebDataset(
        shard_pattern=config['data']['webdataset_shards_train'],
        shuffle=False,  # Disable for testing
        buffer_size=0,
        mode="train"
    )

    print(f"Dataset path: {dataset.shard_pattern}")
    print(f"Num samples: {len(dataset)}")

    # Test configuration
    batch_size = 2
    num_workers = 0  # Single process for debugging
    num_batches = 5

    print(f"\n" + "="*80)
    print(f"Test Configuration")
    print(f"="*80)
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers} (single-process mode)")
    print(f"Num batches to load: {num_batches}")

    # Create DataLoader
    print(f"\nCreating DataLoader...")
    start = time.time()
    dataloader = dataset.create_pytorch_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=2
    )
    print(f"DataLoader created in {time.time()-start:.2f}s")

    # Load batches
    print(f"\n" + "="*80)
    print(f"Loading {num_batches} batches...")
    print(f"="*80 + "\n")

    try:
        batch_times = []

        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            batch_start = time.time()
            volume, report, labels, study_id, embed = batch
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            print(f"Batch {i+1}/{num_batches}:")
            print(f"  Volume shape: {volume.shape}, dtype: {volume.dtype}")
            print(f"  Batch size: {len(study_id)}")
            print(f"  Study IDs: {study_id}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Time: {batch_time:.3f}s")
            print(f"  ✓ Success\n")

        # Summary
        print("="*80)
        print("Test Results")
        print("="*80)
        print(f"✓ Successfully loaded {len(batch_times)} batches")
        print(f"  Mean time per batch: {sum(batch_times)/len(batch_times):.3f}s")
        print(f"  Total time: {sum(batch_times):.3f}s")
        print("="*80)

    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        return 1

    except Exception as e:
        print(f"\n✗ Error during loading:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
