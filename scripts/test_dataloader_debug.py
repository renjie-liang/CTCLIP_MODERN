#!/usr/bin/env python3
"""
Debug script for testing WebDataset DataLoader.

This script tests data loading with detailed timing and error reporting.
"""

import sys
import time
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from src.data.webdataset_loader import CTReportWebDataset
from src.utils.config import load_config


def test_single_sample(dataset, num_samples=5):
    """Test loading single samples without DataLoader."""
    print("="*80)
    print("Test 1: Loading samples without DataLoader (single-process)")
    print("="*80)

    # Create WebDataset pipeline
    import webdataset as wds

    shard_shuffle = 100 if dataset.shuffle else False
    wds_dataset = (
        wds.WebDataset(dataset.shard_pattern, shardshuffle=shard_shuffle, empty_check=False)
        .shuffle(dataset.buffer_size if dataset.shuffle else 0)
        .map(dataset._decode_sample)
    )

    print(f"\nAttempting to load {num_samples} samples...")

    for i, sample in enumerate(wds_dataset):
        if i >= num_samples:
            break

        volume, report, labels, study_id, embed = sample

        print(f"\nSample {i+1}:")
        print(f"  Study ID: {study_id}")
        print(f"  Volume shape: {volume.shape}, dtype: {volume.dtype}")
        print(f"  Report length: {len(report)} chars")
        print(f"  Labels shape: {labels.shape}")
        print(f"  ✓ Successfully loaded")

    print(f"\n✓ Successfully loaded {min(num_samples, i+1)} samples without DataLoader")


def test_dataloader_single_worker(dataset, batch_size=2, num_batches=3):
    """Test DataLoader with num_workers=0 (single process)."""
    print("\n" + "="*80)
    print("Test 2: DataLoader with num_workers=0 (single process)")
    print("="*80)

    print(f"\nCreating DataLoader with batch_size={batch_size}, num_workers=0...")

    start_time = time.time()
    dataloader = dataset.create_pytorch_dataloader(
        batch_size=batch_size,
        num_workers=0,  # Single process - easier to debug
        prefetch_factor=2
    )
    print(f"DataLoader created in {time.time() - start_time:.2f}s")

    print(f"\nLoading {num_batches} batches...")

    try:
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            batch_start = time.time()
            volume, report, labels, study_id, embed = batch
            batch_time = time.time() - batch_start

            print(f"\nBatch {i+1}:")
            print(f"  Volume shape: {volume.shape}, dtype: {volume.dtype}")
            print(f"  Batch size: {len(study_id)}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Time: {batch_time:.3f}s")
            print(f"  ✓ Successfully loaded")

        print(f"\n✓ Successfully loaded {min(num_batches, i+1)} batches with num_workers=0")
        return True

    except Exception as e:
        print(f"\n✗ Error loading batch {i+1}:")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False


def test_dataloader_multi_worker(dataset, batch_size=2, num_workers=4, num_batches=3):
    """Test DataLoader with multiple workers."""
    print("\n" + "="*80)
    print(f"Test 3: DataLoader with num_workers={num_workers} (multi-process)")
    print("="*80)

    print(f"\nCreating DataLoader with batch_size={batch_size}, num_workers={num_workers}...")

    start_time = time.time()
    dataloader = dataset.create_pytorch_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=2
    )
    print(f"DataLoader created in {time.time() - start_time:.2f}s")

    print(f"\nLoading {num_batches} batches...")
    print("NOTE: If this hangs or crashes, there's likely an issue with multi-process loading\n")

    try:
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            batch_start = time.time()
            volume, report, labels, study_id, embed = batch
            batch_time = time.time() - batch_start

            print(f"\nBatch {i+1}:")
            print(f"  Volume shape: {volume.shape}, dtype: {volume.dtype}")
            print(f"  Batch size: {len(study_id)}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Time: {batch_time:.3f}s")
            print(f"  ✓ Successfully loaded")

        print(f"\n✓ Successfully loaded {min(num_batches, i+1)} batches with num_workers={num_workers}")
        return True

    except Exception as e:
        print(f"\n✗ Error loading batch {i+1}:")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False


def benchmark_loading_speed(dataset, num_batches=10, batch_size=4, num_workers=8):
    """Benchmark loading speed."""
    print("\n" + "="*80)
    print(f"Test 4: Benchmark loading speed")
    print("="*80)

    print(f"\nBenchmarking with batch_size={batch_size}, num_workers={num_workers}")

    dataloader = dataset.create_pytorch_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=2
    )

    times = []

    print(f"\nLoading {num_batches} batches...")
    try:
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            start = time.time()
            volume, report, labels, study_id, embed = batch
            elapsed = time.time() - start
            times.append(elapsed)

            if (i + 1) % 5 == 0:
                print(f"  Loaded {i+1}/{num_batches} batches, avg time: {np.mean(times):.3f}s")

        if times:
            print(f"\n✓ Benchmark results:")
            print(f"  Total batches: {len(times)}")
            print(f"  Mean time per batch: {np.mean(times):.3f}s")
            print(f"  Std dev: {np.std(times):.3f}s")
            print(f"  Min: {np.min(times):.3f}s")
            print(f"  Max: {np.max(times):.3f}s")
            print(f"  Throughput: {batch_size / np.mean(times):.2f} samples/sec")

        return True

    except Exception as e:
        print(f"\n✗ Benchmark failed:")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False


def main():
    print("="*80)
    print("WebDataset DataLoader Debug Test")
    print("="*80)

    # Load config
    config_path = project_root / "configs" / "base_config.yaml"
    print(f"\nLoading config from: {config_path}")
    config = load_config(str(config_path))

    # Test training dataset
    print("\n" + "="*80)
    print("Testing TRAINING dataset")
    print("="*80)

    train_dataset = CTReportWebDataset(
        shard_pattern=config['data']['webdataset_shards_train'],
        shuffle=False,  # Disable shuffle for debugging
        buffer_size=0,
        mode="train"
    )

    print(f"Dataset: {train_dataset.shard_pattern}")
    print(f"Num samples: {len(train_dataset)}")

    # Run tests
    results = {}

    # Test 1: Single samples without DataLoader
    try:
        test_single_sample(train_dataset, num_samples=5)
        results['single_sample'] = True
    except Exception as e:
        print(f"\n✗ Test 1 failed: {e}")
        traceback.print_exc()
        results['single_sample'] = False

    # Test 2: DataLoader with num_workers=0
    try:
        results['single_worker'] = test_dataloader_single_worker(
            train_dataset, batch_size=2, num_batches=3
        )
    except Exception as e:
        print(f"\n✗ Test 2 failed: {e}")
        traceback.print_exc()
        results['single_worker'] = False

    # Test 3: DataLoader with num_workers=4
    if results.get('single_worker', False):
        try:
            results['multi_worker'] = test_dataloader_multi_worker(
                train_dataset, batch_size=2, num_workers=4, num_batches=3
            )
        except Exception as e:
            print(f"\n✗ Test 3 failed: {e}")
            traceback.print_exc()
            results['multi_worker'] = False
    else:
        print("\n⚠ Skipping multi-worker test (single-worker test failed)")
        results['multi_worker'] = False

    # Test 4: Benchmark
    if results.get('multi_worker', False):
        try:
            results['benchmark'] = benchmark_loading_speed(
                train_dataset, num_batches=10, batch_size=4, num_workers=8
            )
        except Exception as e:
            print(f"\n✗ Test 4 failed: {e}")
            traceback.print_exc()
            results['benchmark'] = False
    else:
        print("\n⚠ Skipping benchmark test (multi-worker test failed)")
        results['benchmark'] = False

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:20s}: {status}")

    print("\n" + "="*80)
    if all(results.values()):
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed. Check the output above for details.")
    print("="*80)


if __name__ == "__main__":
    main()
