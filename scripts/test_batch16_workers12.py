#!/usr/bin/env python3
"""
Test batch_size=16 with num_workers=12 configuration.

This tests if we can increase batch size by reducing workers,
keeping memory footprint constant.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
import numpy as np
import torch
from src.data.webdataset_loader import CTReportWebDataset
from src.utils.config import load_config


def main():
    print("="*80)
    print("Test: batch_size=16, num_workers=12")
    print("="*80)
    print("\nRationale:")
    print("  Current:    batch_size=8  × num_workers=24 × prefetch=2 × 2GB = 768GB")
    print("  Proposed:   batch_size=16 × num_workers=12 × prefetch=2 × 2GB = 768GB")
    print("  Benefits:   Better GPU utilization, more efficient gradients")
    print("="*80)

    # Load config
    config_path = project_root / "configs" / "base_config.yaml"
    config = load_config(str(config_path))

    # Create dataset
    print("\nCreating dataset...")
    dataset = CTReportWebDataset(
        shard_pattern=config['data']['webdataset_shards_train'],
        shuffle=False,
        buffer_size=0,
        mode="train"
    )

    print(f"Dataset: {len(dataset)} samples")

    # Create DataLoader with new config
    print("\n" + "="*80)
    print("Creating DataLoader (batch_size=16, num_workers=12)...")
    print("="*80)

    try:
        start = time.time()
        dataloader = dataset.create_pytorch_dataloader(
            batch_size=16,
            num_workers=12,
            prefetch_factor=2
        )
        creation_time = time.time() - start
        print(f"✓ DataLoader created in {creation_time:.2f}s")

    except Exception as e:
        print(f"✗ Failed to create DataLoader: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Load batches
    print("\n" + "="*80)
    print("Loading 20 batches...")
    print("="*80)

    try:
        batch_times = []
        total_samples = 0
        start = time.time()

        for i, batch in enumerate(dataloader):
            if i >= 20:
                break

            batch_start = time.time()
            volume, report, labels, study_id, embed = batch
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            total_samples += len(study_id)

            # Print progress
            avg_time = np.mean(batch_times)
            throughput = 16 / avg_time if avg_time > 0 else 0

            if (i + 1) % 5 == 0:
                print(f"Batch {i+1:2d}/20: "
                      f"{batch_time:.3f}s | "
                      f"avg: {avg_time:.3f}s | "
                      f"throughput: {throughput:6.2f} samples/sec | "
                      f"shape: {volume.shape}")

        total_time = time.time() - start

        # Summary
        print("\n" + "="*80)
        print("Results")
        print("="*80)
        print(f"✓ Successfully loaded {len(batch_times)} batches")
        print(f"\nStatistics:")
        print(f"  Total samples: {total_samples}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Mean time per batch: {np.mean(batch_times):.3f}s ± {np.std(batch_times):.3f}s")
        print(f"  Throughput: {total_samples / total_time:.2f} samples/sec")

        # Compare with current config
        print("\n" + "="*80)
        print("Comparison with Current Config")
        print("="*80)
        current_throughput = 2.4  # Approximate from training logs
        new_throughput = total_samples / total_time
        speedup = new_throughput / current_throughput

        print(f"  Current (batch=8, workers=24):  ~{current_throughput:.1f} samples/sec")
        print(f"  New (batch=16, workers=12):     {new_throughput:.2f} samples/sec")
        print(f"  Speedup: {speedup:.2f}x")

        if speedup > 1.0:
            print(f"\n✓ Configuration IMPROVED throughput!")
            print(f"  Recommendation: Update config to batch_size=16, num_workers=12")
        else:
            print(f"\n⚠ Configuration did not improve throughput")
            print(f"  Recommendation: Keep current config (batch_size=8, num_workers=24)")

        print("="*80)
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        return 1

    except Exception as e:
        print(f"\n✗ Error during loading: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
