#!/usr/bin/env python3
"""
Test a specific batch_size and num_workers configuration.

Usage:
    python scripts/test_specific_config.py --batch_size 32 --num_workers 16
    python scripts/test_specific_config.py --batch_size 16 --num_workers 8 --num_batches 50
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
import numpy as np
from src.data.webdataset_loader import CTReportWebDataset
from src.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Test DataLoader configuration')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers')
    parser.add_argument('--num_batches', type=int, default=20, help='Number of batches to test')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Prefetch factor')
    return parser.parse_args()


def main():
    args = parse_args()

    print("="*80)
    print("DataLoader Configuration Test")
    print("="*80)

    # Load config
    config_path = project_root / "configs" / "base_config.yaml"
    config = load_config(str(config_path))

    # Create dataset
    print(f"\nCreating dataset...")
    dataset = CTReportWebDataset(
        shard_pattern=config['data']['webdataset_shards_train'],
        shuffle=False,
        buffer_size=0,
        mode="train"
    )

    print(f"Dataset: {len(dataset)} samples")

    # Configuration
    print("\n" + "="*80)
    print("Configuration")
    print("="*80)
    print(f"Batch size: {args.batch_size}")
    print(f"Num workers: {args.num_workers}")
    print(f"Prefetch factor: {args.prefetch_factor}")
    print(f"Batches to load: {args.num_batches}")

    # Estimate memory usage
    single_sample_gb = 2.0  # Rough estimate for 480x480x240 float32 + intermediate data
    estimated_memory_gb = args.num_workers * args.prefetch_factor * args.batch_size * single_sample_gb
    print(f"\n⚠ Estimated memory usage: {estimated_memory_gb:.1f} GB")
    if estimated_memory_gb > 500:
        print(f"   WARNING: This may exceed available memory!")

    input("\nPress Enter to continue...")

    # Create DataLoader
    print("\n" + "="*80)
    print("Creating DataLoader...")
    print("="*80)

    try:
        start = time.time()
        dataloader = dataset.create_pytorch_dataloader(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor
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
    print(f"Loading {args.num_batches} batches...")
    print("="*80)

    try:
        batch_times = []
        total_samples = 0
        start = time.time()

        for i, batch in enumerate(dataloader):
            if i >= args.num_batches:
                break

            batch_start = time.time()
            volume, report, labels, study_id, embed = batch
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            total_samples += len(study_id)

            # Print progress
            avg_time = np.mean(batch_times)
            throughput = args.batch_size / avg_time if avg_time > 0 else 0

            print(f"Batch {i+1:3d}/{args.num_batches}: "
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
        print(f"  Min batch time: {np.min(batch_times):.3f}s")
        print(f"  Max batch time: {np.max(batch_times):.3f}s")
        print(f"  Throughput: {total_samples / total_time:.2f} samples/sec")
        print(f"  Time per sample: {total_time / total_samples:.3f}s")

        # Estimate full epoch time
        total_samples_in_dataset = len(dataset)
        estimated_epoch_time = (total_samples_in_dataset / args.batch_size) * np.mean(batch_times)
        print(f"\nEstimates:")
        print(f"  Time per epoch: {estimated_epoch_time / 3600:.2f} hours")
        print(f"  Batches per epoch: {total_samples_in_dataset // args.batch_size}")

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
