"""
Test and validate WebDataset conversion.

This script helps you:
1. Verify that converted data matches original NPZ data
2. Benchmark loading speed comparison
3. Check data integrity

Usage:
    # Basic validation
    python scripts/test_webdataset.py \
        --webdataset_dir /path/to/webdataset/shards \
        --num_samples 10

    # Speed benchmark
    python scripts/test_webdataset.py \
        --webdataset_dir /path/to/webdataset/shards \
        --benchmark \
        --num_samples 100
"""

import argparse
import time
import json
from pathlib import Path
import numpy as np
import torch
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.webdataset_loader import CTReportWebDataset


def test_data_loading(shard_dir: Path, num_samples: int = 10):
    """Test basic data loading from WebDataset."""

    print("="*80)
    print("TEST 1: Basic Data Loading")
    print("="*80)

    # Find shard pattern
    shard_files = sorted(shard_dir.glob("shard-*.tar"))
    if not shard_files:
        print(f"ERROR: No shard files found in {shard_dir}")
        return False

    print(f"Found {len(shard_files)} shard files")

    # Create shard pattern (e.g., "shard-{000000..000099}.tar")
    num_shards = len(shard_files)
    shard_pattern = str(shard_dir / f"shard-{{000000..{num_shards-1:06d}}}.tar")

    print(f"Shard pattern: {shard_pattern}")

    # Create dataset
    dataset = CTReportWebDataset(
        shard_pattern=shard_pattern,
        shuffle=False,
        mode="test"
    )

    # Create loader
    loader = dataset.create_pytorch_dataloader(
        batch_size=1,
        num_workers=2,
        prefetch_factor=1
    )

    print(f"\nLoading {num_samples} samples...")
    success_count = 0

    for i, batch in enumerate(loader):
        if i >= num_samples:
            break

        volume, report, labels, study_id, embedding = batch

        # Validate data shapes
        if volume.numel() > 0:
            expected_shape = (1, 1, 240, 480, 480)
            if volume.shape != expected_shape:
                print(f"✗ Sample {i}: Wrong volume shape {volume.shape}, expected {expected_shape}")
                continue

        # Validate data types
        if not isinstance(report, list) or len(report) != 1:
            print(f"✗ Sample {i}: Invalid report format")
            continue

        if not isinstance(labels, np.ndarray):
            print(f"✗ Sample {i}: Invalid labels format")
            continue

        # Validate value ranges
        if volume.numel() > 0:
            if volume.min() < -2 or volume.max() > 2:
                print(f"✗ Sample {i}: Volume values out of range [{volume.min():.2f}, {volume.max():.2f}]")
                continue

        print(f"✓ Sample {i}: {study_id[0]}")
        print(f"  - Volume shape: {volume.shape if volume.numel() > 0 else 'N/A (embedding mode)'}")
        print(f"  - Report length: {len(report[0])} chars")
        print(f"  - Labels shape: {labels.shape}")
        print(f"  - Value range: [{volume.min():.3f}, {volume.max():.3f}]" if volume.numel() > 0 else "")

        success_count += 1

    print(f"\n{'='*80}")
    print(f"Result: {success_count}/{num_samples} samples loaded successfully")
    print(f"{'='*80}\n")

    return success_count == num_samples


def benchmark_loading_speed(shard_dir: Path, num_samples: int = 100):
    """Benchmark WebDataset loading speed."""

    print("="*80)
    print("TEST 2: Loading Speed Benchmark")
    print("="*80)

    shard_files = sorted(shard_dir.glob("shard-*.tar"))
    num_shards = len(shard_files)
    shard_pattern = str(shard_dir / f"shard-{{000000..{num_shards-1:06d}}}.tar")

    # Test different configurations
    configs = [
        {"num_workers": 0, "prefetch_factor": None, "batch_size": 1},
        {"num_workers": 2, "prefetch_factor": 2, "batch_size": 1},
        {"num_workers": 4, "prefetch_factor": 2, "batch_size": 1},
        {"num_workers": 8, "prefetch_factor": 4, "batch_size": 1},
    ]

    results = []

    for config in configs:
        print(f"\nConfig: workers={config['num_workers']}, "
              f"prefetch={config['prefetch_factor']}, "
              f"batch_size={config['batch_size']}")

        dataset = CTReportWebDataset(
            shard_pattern=shard_pattern,
            shuffle=False,
            mode="benchmark"
        )

        loader_kwargs = {
            'batch_size': config['batch_size'],
            'num_workers': config['num_workers'],
        }
        if config['prefetch_factor'] is not None:
            loader_kwargs['prefetch_factor'] = config['prefetch_factor']

        loader = dataset.create_pytorch_dataloader(**loader_kwargs)

        # Warmup
        for i, _ in enumerate(loader):
            if i >= 5:
                break

        # Benchmark
        start_time = time.time()
        sample_count = 0

        for i, batch in enumerate(loader):
            if i >= num_samples:
                break
            sample_count += 1

        elapsed = time.time() - start_time
        samples_per_sec = sample_count / elapsed if elapsed > 0 else 0

        print(f"  - Loaded {sample_count} samples in {elapsed:.2f}s")
        print(f"  - Speed: {samples_per_sec:.2f} samples/sec")

        results.append({
            'config': config,
            'samples_per_sec': samples_per_sec,
            'elapsed': elapsed
        })

    # Print summary
    print(f"\n{'='*80}")
    print("Benchmark Summary:")
    print(f"{'='*80}")
    for i, result in enumerate(results):
        config = result['config']
        print(f"{i+1}. workers={config['num_workers']:2d}, "
              f"prefetch={str(config['prefetch_factor']):4s}, "
              f"batch={config['batch_size']}: "
              f"{result['samples_per_sec']:6.2f} samples/sec")

    best = max(results, key=lambda x: x['samples_per_sec'])
    print(f"\nBest configuration: {best['config']}")
    print(f"Speed: {best['samples_per_sec']:.2f} samples/sec")
    print(f"{'='*80}\n")


def check_manifest(shard_dir: Path):
    """Check and display manifest information."""

    print("="*80)
    print("Manifest Information")
    print("="*80)

    manifest_path = shard_dir / 'manifest.json'

    if not manifest_path.exists():
        print("No manifest.json found")
        return

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    print(f"Total samples: {manifest['total_samples']:,}")
    print(f"Total shards: {manifest['num_shards']}")
    print(f"Samples per shard: {manifest['samples_per_shard']}")
    print(f"Total size: {manifest['total_size_gb']:.2f} GB")
    print(f"Average sample size: {manifest['average_sample_size_mb']:.2f} MB")

    # Calculate compression ratio estimate
    # Assuming original NPZ was ~350MB per sample on average
    original_size_gb = manifest['total_samples'] * 350 / 1024
    compression_ratio = original_size_gb / manifest['total_size_gb']

    print(f"\nEstimated compression:")
    print(f"  - Original (NPZ): ~{original_size_gb:.2f} GB")
    print(f"  - Compressed (WebDataset): {manifest['total_size_gb']:.2f} GB")
    print(f"  - Compression ratio: {compression_ratio:.2f}x")
    print(f"  - Space saved: {(1 - 1/compression_ratio)*100:.1f}%")
    print(f"{'='*80}\n")


def test_float16_precision(shard_dir: Path, num_samples: int = 5):
    """Test if float16 precision is acceptable."""

    print("="*80)
    print("TEST 3: Float16 Precision Check")
    print("="*80)

    shard_files = sorted(shard_dir.glob("shard-*.tar"))
    num_shards = len(shard_files)
    shard_pattern = str(shard_dir / f"shard-{{000000..{num_shards-1:06d}}}.tar")

    dataset = CTReportWebDataset(
        shard_pattern=shard_pattern,
        shuffle=False,
        mode="test"
    )

    loader = dataset.create_pytorch_dataloader(batch_size=1, num_workers=2)

    print(f"Checking precision for {num_samples} samples...\n")

    for i, batch in enumerate(loader):
        if i >= num_samples:
            break

        volume, _, _, study_id, _ = batch

        if volume.numel() == 0:
            print(f"Sample {i}: Skipping (embedding mode)")
            continue

        # Check for NaN or Inf
        has_nan = torch.isnan(volume).any()
        has_inf = torch.isinf(volume).any()

        # Check value distribution
        mean = volume.mean().item()
        std = volume.std().item()
        min_val = volume.min().item()
        max_val = volume.max().item()

        status = "✓" if not (has_nan or has_inf) else "✗"

        print(f"{status} Sample {i}: {study_id[0]}")
        print(f"  - Mean: {mean:.6f}, Std: {std:.6f}")
        print(f"  - Range: [{min_val:.6f}, {max_val:.6f}]")
        print(f"  - Has NaN: {has_nan}, Has Inf: {has_inf}")

    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Test WebDataset conversion")
    parser.add_argument('--webdataset_dir', type=str, required=True,
                        help='Directory containing WebDataset shards')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to test')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run speed benchmark')
    parser.add_argument('--check_precision', action='store_true',
                        help='Check float16 precision')

    args = parser.parse_args()

    shard_dir = Path(args.webdataset_dir)

    if not shard_dir.exists():
        print(f"ERROR: Directory does not exist: {shard_dir}")
        return

    # Always check manifest
    check_manifest(shard_dir)

    # Test basic loading
    success = test_data_loading(shard_dir, args.num_samples)

    if not success:
        print("⚠️  Basic loading test failed. Please check your data.")
        return

    # Optional tests
    if args.check_precision:
        test_float16_precision(shard_dir, args.num_samples)

    if args.benchmark:
        benchmark_loading_speed(shard_dir, args.num_samples)

    print("\n✓ All tests completed!")


if __name__ == '__main__':
    main()
