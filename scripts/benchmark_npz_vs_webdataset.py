"""
Benchmark NPZ vs WebDataset loading speed.

This script compares the I/O performance of:
1. Original NPZ format (with mmap_mode)
2. WebDataset format (float16)

It measures both raw I/O time and total loading time (I/O + processing).

Usage:
    python scripts/benchmark_npz_vs_webdataset.py \
        --npz_dir /path/to/npz/files \
        --webdataset_dir /path/to/webdataset/shards \
        --reports_file /path/to/reports.csv \
        --meta_file /path/to/metadata.csv \
        --labels_file /path/to/labels.csv \
        --num_samples 100
"""

import argparse
import time
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.npz_loader import CTReportNPZDataset
from data.webdataset_loader import CTReportWebDataset


def benchmark_npz_raw_io(npz_files, num_samples=100):
    """Benchmark raw NPZ file I/O (no processing)."""
    print("\n" + "="*80)
    print("NPZ - Raw I/O Benchmark (just np.load)")
    print("="*80)

    npz_files = npz_files[:num_samples]

    times = []
    for npz_path in tqdm(npz_files, desc="Loading NPZ"):
        start = time.time()
        data = np.load(npz_path, mmap_mode='r')["volume"]
        _ = data.shape  # Force read metadata
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = np.mean(times)
    total_time = np.sum(times)

    print(f"Samples loaded: {len(npz_files)}")
    print(f"Average time per sample: {avg_time*1000:.2f} ms")
    print(f"Total time: {total_time:.2f} s")
    print(f"Throughput: {len(npz_files)/total_time:.2f} samples/sec")

    return avg_time, total_time


def benchmark_npz_full_loading(data_cfg, mode='train', num_samples=100):
    """Benchmark full NPZ loading with CTReportNPZDataset (I/O + processing)."""
    print("\n" + "="*80)
    print("NPZ - Full Loading Benchmark (I/O + processing)")
    print("="*80)

    dataset = CTReportNPZDataset(
        data_folder=data_cfg['npz_dir'],
        reports_file=data_cfg['reports_file'],
        meta_file=data_cfg['meta_file'],
        labels_file=data_cfg['labels_file'],
        mode=mode
    )

    # Limit samples
    dataset.samples = dataset.samples[:num_samples]

    times = []
    for i in tqdm(range(min(num_samples, len(dataset))), desc="Loading with processing"):
        start = time.time()
        _ = dataset[i]
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = np.mean(times)
    total_time = np.sum(times)

    print(f"Samples loaded: {len(times)}")
    print(f"Average time per sample: {avg_time*1000:.2f} ms")
    print(f"Total time: {total_time:.2f} s")
    print(f"Throughput: {len(times)/total_time:.2f} samples/sec")

    return avg_time, total_time


def benchmark_webdataset_loading(shard_dir, num_samples=100):
    """Benchmark WebDataset loading (I/O + processing)."""
    print("\n" + "="*80)
    print("WebDataset - Full Loading Benchmark (I/O + processing)")
    print("="*80)

    shard_files = sorted(shard_dir.glob("shard-*.tar"))
    num_shards = len(shard_files)
    shard_pattern = str(shard_dir / f"shard-{{000000..{num_shards-1:06d}}}.tar")

    dataset = CTReportWebDataset(
        shard_pattern=shard_pattern,
        shuffle=False,
        mode="benchmark"
    )

    # Create loader with single worker for fair comparison
    loader = dataset.create_pytorch_dataloader(
        batch_size=1,
        num_workers=0
    )

    times = []
    count = 0

    for batch in tqdm(loader, desc="Loading WebDataset", total=num_samples):
        start = time.time()
        # Data is already loaded, just access it
        volume, report, labels, study_id, embedding = batch
        _ = volume.shape if volume.numel() > 0 else None
        elapsed = time.time() - start
        times.append(elapsed)

        count += 1
        if count >= num_samples:
            break

    avg_time = np.mean(times)
    total_time = np.sum(times)

    print(f"Samples loaded: {len(times)}")
    print(f"Average time per sample: {avg_time*1000:.2f} ms")
    print(f"Total time: {total_time:.2f} s")
    print(f"Throughput: {len(times)/total_time:.2f} samples/sec")

    return avg_time, total_time


def benchmark_npz_file_sizes(npz_files, num_samples=100):
    """Measure NPZ file sizes."""
    npz_files = npz_files[:num_samples]

    sizes = []
    for npz_path in npz_files:
        size_mb = Path(npz_path).stat().st_size / (1024**2)
        sizes.append(size_mb)

    total_size_gb = sum(sizes) / 1024
    avg_size_mb = np.mean(sizes)

    return avg_size_mb, total_size_gb


def main():
    parser = argparse.ArgumentParser(description="Benchmark NPZ vs WebDataset")
    parser.add_argument('--npz_dir', type=str, required=True,
                        help='Directory containing NPZ files')
    parser.add_argument('--webdataset_dir', type=str, required=True,
                        help='Directory containing WebDataset shards')
    parser.add_argument('--reports_file', type=str, required=True,
                        help='CSV file with reports')
    parser.add_argument('--meta_file', type=str, required=True,
                        help='CSV file with metadata')
    parser.add_argument('--labels_file', type=str, required=True,
                        help='CSV file with labels')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to benchmark')

    args = parser.parse_args()

    print("="*80)
    print("NPZ vs WebDataset Loading Benchmark")
    print("="*80)
    print(f"NPZ directory: {args.npz_dir}")
    print(f"WebDataset directory: {args.webdataset_dir}")
    print(f"Samples to test: {args.num_samples}")

    # Find NPZ files
    print("\nScanning NPZ files...")
    npz_files_3layer = glob.glob(f"{args.npz_dir}/*/*/*npz")
    npz_files_2layer = glob.glob(f"{args.npz_dir}/*/*npz")
    npz_files_1layer = glob.glob(f"{args.npz_dir}/*npz")

    npz_files = max([npz_files_3layer, npz_files_2layer, npz_files_1layer],
                     key=len)

    print(f"Found {len(npz_files)} NPZ files")

    if len(npz_files) == 0:
        print("ERROR: No NPZ files found!")
        return

    # Measure file sizes
    print("\n" + "="*80)
    print("File Size Comparison")
    print("="*80)

    npz_avg_mb, npz_total_gb = benchmark_npz_file_sizes(npz_files, args.num_samples)
    print(f"NPZ - Average file size: {npz_avg_mb:.2f} MB")
    print(f"NPZ - Total size ({args.num_samples} samples): {npz_total_gb:.2f} GB")

    # Check WebDataset size
    shard_dir = Path(args.webdataset_dir)
    manifest_path = shard_dir / 'manifest.json'
    if manifest_path.exists():
        import json
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        wd_avg_mb = manifest['average_sample_size_mb']
        wd_total_gb = (wd_avg_mb * args.num_samples) / 1024
        print(f"WebDataset - Average sample size: {wd_avg_mb:.2f} MB")
        print(f"WebDataset - Estimated size ({args.num_samples} samples): {wd_total_gb:.2f} GB")
        print(f"\nCompression ratio: {npz_avg_mb / wd_avg_mb:.2f}x")
        print(f"Space savings: {(1 - wd_avg_mb/npz_avg_mb)*100:.1f}%")

    # Benchmark 1: NPZ raw I/O
    npz_io_avg, npz_io_total = benchmark_npz_raw_io(npz_files, args.num_samples)

    # Benchmark 2: NPZ full loading
    data_cfg = {
        'npz_dir': args.npz_dir,
        'reports_file': args.reports_file,
        'meta_file': args.meta_file,
        'labels_file': args.labels_file
    }
    npz_full_avg, npz_full_total = benchmark_npz_full_loading(
        data_cfg, mode='val', num_samples=args.num_samples
    )

    # Benchmark 3: WebDataset loading
    wd_avg, wd_total = benchmark_webdataset_loading(
        shard_dir, num_samples=args.num_samples
    )

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\n{'Method':<30} {'Avg Time (ms)':<15} {'Throughput (samples/s)':<25}")
    print("-"*70)
    print(f"{'NPZ (raw I/O only)':<30} {npz_io_avg*1000:>14.2f} {args.num_samples/npz_io_total:>24.2f}")
    print(f"{'NPZ (full loading)':<30} {npz_full_avg*1000:>14.2f} {args.num_samples/npz_full_total:>24.2f}")
    print(f"{'WebDataset (full loading)':<30} {wd_avg*1000:>14.2f} {args.num_samples/wd_total:>24.2f}")

    # Calculate speedups
    print(f"\n{'Comparison':<30} {'Speedup':<15}")
    print("-"*45)
    if npz_full_total > 0:
        speedup = npz_full_total / wd_total
        print(f"{'WebDataset vs NPZ (full)':<30} {speedup:>14.2f}x")

    # Processing overhead
    print(f"\n{'Processing Overhead Analysis':<30}")
    print("-"*45)
    processing_time_npz = npz_full_avg - npz_io_avg
    print(f"NPZ I/O time: {npz_io_avg*1000:.2f} ms")
    print(f"NPZ processing time: {processing_time_npz*1000:.2f} ms")
    print(f"NPZ processing overhead: {(processing_time_npz/npz_full_avg)*100:.1f}%")

    print("\n" + "="*80)
    print("Conclusion:")
    if speedup > 1.5:
        print(f"✓ WebDataset is {speedup:.2f}x FASTER than NPZ!")
    elif speedup > 1.0:
        print(f"✓ WebDataset is {speedup:.2f}x faster (modest improvement)")
    else:
        print(f"✗ WebDataset is {1/speedup:.2f}x SLOWER (investigation needed)")
        print("\nPossible reasons:")
        print("  - Processing overhead dominates I/O time")
        print("  - Need more workers for WebDataset to show advantage")
        print("  - Small test set doesn't show sequential I/O benefits")
        print("  - Try benchmarking with more samples (500-1000)")
    print("="*80)


if __name__ == '__main__':
    main()
