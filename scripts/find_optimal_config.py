#!/usr/bin/env python3
"""
Find optimal num_workers and batch_size configuration.

This script tests different combinations to find the best performance.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
import numpy as np
from src.data.webdataset_loader import CTReportWebDataset
from src.utils.config import load_config


def test_configuration(dataset, batch_size, num_workers, num_batches=20):
    """
    Test a specific configuration and measure throughput.

    Returns:
        dict: Results including throughput, mean time, success status
    """
    print(f"\n{'='*80}")
    print(f"Testing: batch_size={batch_size}, num_workers={num_workers}")
    print(f"{'='*80}")

    try:
        # Create DataLoader
        start = time.time()
        dataloader = dataset.create_pytorch_dataloader(
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=2 if num_workers > 0 else 2
        )
        loader_creation_time = time.time() - start
        print(f"DataLoader creation time: {loader_creation_time:.2f}s")

        # Load batches
        batch_times = []
        total_samples = 0

        print(f"Loading {num_batches} batches...")

        start = time.time()
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            batch_start = time.time()
            volume, report, labels, study_id, embed = batch
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            total_samples += len(study_id)

            if (i + 1) % 5 == 0:
                avg_time = np.mean(batch_times)
                throughput = batch_size / avg_time if avg_time > 0 else 0
                print(f"  Batch {i+1}/{num_batches}: {batch_time:.3f}s/batch, "
                      f"avg: {avg_time:.3f}s/batch, "
                      f"throughput: {throughput:.2f} samples/sec")

        total_time = time.time() - start

        # Calculate metrics
        mean_time = np.mean(batch_times) if batch_times else 0
        std_time = np.std(batch_times) if batch_times else 0
        throughput = total_samples / total_time if total_time > 0 else 0

        print(f"\n✓ Success:")
        print(f"  Total samples: {total_samples}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Mean time per batch: {mean_time:.3f}s ± {std_time:.3f}s")
        print(f"  Throughput: {throughput:.2f} samples/sec")

        return {
            'success': True,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'mean_time': mean_time,
            'std_time': std_time,
            'throughput': throughput,
            'total_time': total_time,
            'total_samples': total_samples
        }

    except KeyboardInterrupt:
        print("\n⚠ Test interrupted by user")
        return None

    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()

        return {
            'success': False,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'error': str(e)
        }


def main():
    print("="*80)
    print("Optimal Configuration Finder")
    print("="*80)

    # Load config
    config_path = project_root / "configs" / "base_config.yaml"
    config = load_config(str(config_path))

    # Create dataset
    print(f"\nCreating dataset...")
    dataset = CTReportWebDataset(
        shard_pattern=config['data']['webdataset_shards_train'],
        shuffle=False,  # Disable for consistent benchmarking
        buffer_size=0,
        mode="train"
    )

    print(f"Dataset: {len(dataset)} samples")

    # Test configurations
    print("\n" + "="*80)
    print("Test Plan")
    print("="*80)
    print("We will test progressively larger configurations:")
    print("1. Single worker with different batch sizes")
    print("2. Multiple workers with optimal batch size")
    print("3. Find the sweet spot")
    print("\nPress Ctrl+C to skip a test if it hangs or is too slow")
    print("="*80)

    input("\nPress Enter to start testing...")

    results = []

    # Stage 1: Test batch sizes with num_workers=0
    print("\n" + "="*80)
    print("Stage 1: Finding optimal batch_size (num_workers=0)")
    print("="*80)

    batch_sizes = [4, 8, 16, 24, 32]

    for batch_size in batch_sizes:
        result = test_configuration(dataset, batch_size=batch_size, num_workers=0, num_batches=10)
        if result is None:  # User interrupted
            break
        results.append(result)

        if not result['success']:
            print(f"\n⚠ batch_size={batch_size} failed, stopping batch size tests")
            break

    # Find best batch size from Stage 1
    successful_results = [r for r in results if r.get('success', False)]
    if successful_results:
        best_batch_result = max(successful_results, key=lambda x: x['throughput'])
        best_batch_size = best_batch_result['batch_size']
        print(f"\n✓ Best batch_size from Stage 1: {best_batch_size} "
              f"(throughput: {best_batch_result['throughput']:.2f} samples/sec)")
    else:
        print("\n✗ No successful configurations in Stage 1, stopping")
        return

    # Stage 2: Test num_workers with best batch size
    print("\n" + "="*80)
    print(f"Stage 2: Finding optimal num_workers (batch_size={best_batch_size})")
    print("="*80)

    num_workers_list = [4, 8, 16, 24, 32]

    for num_workers in num_workers_list:
        result = test_configuration(dataset, batch_size=best_batch_size, num_workers=num_workers, num_batches=20)
        if result is None:  # User interrupted
            break
        results.append(result)

        if not result['success']:
            print(f"\n⚠ num_workers={num_workers} failed")
            # Continue testing - maybe higher values work

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)

    successful_results = [r for r in results if r.get('success', False)]

    if not successful_results:
        print("✗ No successful configurations found")
        return

    # Sort by throughput
    successful_results.sort(key=lambda x: x['throughput'], reverse=True)

    print("\nTop 5 Configurations (by throughput):")
    print(f"{'Rank':<6} {'Batch':<8} {'Workers':<10} {'Throughput':<20} {'Mean Time':<15}")
    print("-" * 80)

    for i, result in enumerate(successful_results[:5], 1):
        print(f"{i:<6} {result['batch_size']:<8} {result['num_workers']:<10} "
              f"{result['throughput']:.2f} samples/sec{'':<6} "
              f"{result['mean_time']:.3f}s ± {result['std_time']:.3f}s")

    # Recommend configuration
    best_result = successful_results[0]
    print("\n" + "="*80)
    print("Recommended Configuration")
    print("="*80)
    print(f"batch_size: {best_result['batch_size']}")
    print(f"num_workers: {best_result['num_workers']}")
    print(f"Expected throughput: {best_result['throughput']:.2f} samples/sec")
    print(f"Expected time per batch: {best_result['mean_time']:.3f}s")
    print("\nUpdate your config file:")
    print("```yaml")
    print("data:")
    print(f"  batch_size: {best_result['batch_size']}")
    print(f"  num_workers: {best_result['num_workers']}")
    print(f"  prefetch_factor: 2")
    print("```")
    print("="*80)


if __name__ == "__main__":
    main()
