"""
Quick comparison script for NPZ vs WebDataset loading speed.

This is a simplified version for rapid testing and comparison.

Usage:
    python scripts/quick_compare_npz_vs_webdataset.py --num_samples 50
"""

import sys
import time
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.npz_loader import CTReportNPZDataset
from data.webdataset_loader import CTReportWebDataset


def benchmark_loader(dataset, num_samples=50, batch_size=1, num_workers=0, name="Loader"):
    """Benchmark a data loader."""
    print(f"\n{'='*80}")
    print(f"Benchmarking: {name}")
    print(f"{'='*80}")
    print(f"  Samples: {num_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Workers: {num_workers}")

    # Create DataLoader
    if hasattr(dataset, 'create_pytorch_dataloader'):
        # WebDataset
        loader = dataset.create_pytorch_dataloader(
            batch_size=batch_size,
            num_workers=num_workers
        )
    else:
        # NPZ Dataset
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    # Benchmark
    times = []
    count = 0

    print("\nLoading samples...")
    start_total = time.time()

    for batch_idx, batch in enumerate(loader):
        batch_start = time.time()

        # Access data (forces loading)
        if isinstance(batch, (list, tuple)):
            volumes = batch[0]
        else:
            volumes = batch

        # Ensure data is on CPU (for fair comparison)
        if torch.is_tensor(volumes) and volumes.numel() > 0:
            _ = volumes.cpu()

        batch_time = time.time() - batch_start
        times.append(batch_time)

        count += batch_size
        if count >= num_samples:
            break

    total_time = time.time() - start_total

    # Statistics
    avg_batch_time = sum(times) / len(times) if times else 0
    avg_sample_time = avg_batch_time / batch_size if batch_size > 0 else 0
    throughput = count / total_time if total_time > 0 else 0

    print(f"\nResults:")
    print(f"  ✓ Total samples loaded: {count}")
    print(f"  ✓ Total time: {total_time:.2f} s")
    print(f"  ✓ Avg time per batch: {avg_batch_time*1000:.2f} ms")
    print(f"  ✓ Avg time per sample: {avg_sample_time*1000:.2f} ms")
    print(f"  ✓ Throughput: {throughput:.2f} samples/sec")

    return {
        'total_time': total_time,
        'avg_sample_time': avg_sample_time,
        'throughput': throughput,
        'num_samples': count
    }


def main():
    parser = argparse.ArgumentParser(description="Quick NPZ vs WebDataset comparison")
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of samples to benchmark')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for DataLoader')
    parser.add_argument('--npz_dir', type=str,
                        default='/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/vaild_npz',
                        help='NPZ directory')
    parser.add_argument('--webdataset_dir', type=str,
                        default='/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/webdataset_val',
                        help='WebDataset directory')

    args = parser.parse_args()

    print("="*80)
    print("NPZ vs WebDataset Quick Comparison")
    print("="*80)
    print(f"Configuration:")
    print(f"  Samples: {args.num_samples}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Workers: {args.num_workers}")

    # ========================================================================
    # Test 1: NPZ DataLoader
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 1: NPZ DataLoader")
    print("="*80)

    try:
        npz_dataset = CTReportNPZDataset(
            data_folder=args.npz_dir,
            reports_file="/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/radiology_text_reports/validation_reports.csv",
            meta_file="/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/metadata/validation_metadata.csv",
            labels_file="/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/multi_abnormality_labels/valid_predicted_labels.csv",
            mode="val"
        )

        npz_results = benchmark_loader(
            npz_dataset,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            name="NPZ DataLoader"
        )
    except Exception as e:
        print(f"\n❌ NPZ DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        npz_results = None

    # ========================================================================
    # Test 2: WebDataset DataLoader
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 2: WebDataset DataLoader")
    print("="*80)

    try:
        # Find shard pattern
        webdataset_path = Path(args.webdataset_dir)
        if not webdataset_path.exists():
            print(f"❌ WebDataset directory not found: {args.webdataset_dir}")
            print("   Skipping WebDataset test...")
            wd_results = None
        else:
            shard_files = sorted(webdataset_path.glob("shard-*.tar"))
            if not shard_files:
                print(f"❌ No shard files found in {args.webdataset_dir}")
                print("   Skipping WebDataset test...")
                wd_results = None
            else:
                num_shards = len(shard_files)
                shard_pattern = str(webdataset_path / f"shard-{{000000..{num_shards-1:06d}}}.tar")

                wd_dataset = CTReportWebDataset(
                    shard_pattern=shard_pattern,
                    shuffle=False,
                    mode="val"
                )

                wd_results = benchmark_loader(
                    wd_dataset,
                    num_samples=args.num_samples,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    name="WebDataset DataLoader"
                )
    except Exception as e:
        print(f"\n❌ WebDataset test failed: {e}")
        import traceback
        traceback.print_exc()
        wd_results = None

    # ========================================================================
    # Summary & Comparison
    # ========================================================================
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    if npz_results and wd_results:
        print(f"\n{'Metric':<30} {'NPZ':<20} {'WebDataset':<20}")
        print("-"*70)
        print(f"{'Avg time per sample (ms)':<30} {npz_results['avg_sample_time']*1000:>19.2f} {wd_results['avg_sample_time']*1000:>19.2f}")
        print(f"{'Throughput (samples/sec)':<30} {npz_results['throughput']:>19.2f} {wd_results['throughput']:>19.2f}")
        print(f"{'Total time (s)':<30} {npz_results['total_time']:>19.2f} {wd_results['total_time']:>19.2f}")

        # Speedup
        speedup = npz_results['total_time'] / wd_results['total_time']
        print(f"\n{'Speedup (WebDataset vs NPZ)':<30} {speedup:>19.2f}x")

        if speedup > 1.2:
            print(f"\n✓ WebDataset is {speedup:.2f}x FASTER than NPZ!")
        elif speedup > 0.8:
            print(f"\n≈ NPZ and WebDataset have similar performance ({speedup:.2f}x)")
        else:
            print(f"\n✓ NPZ is {1/speedup:.2f}x FASTER than WebDataset!")

    elif npz_results:
        print("\nNPZ Results:")
        print(f"  Avg time: {npz_results['avg_sample_time']*1000:.2f} ms/sample")
        print(f"  Throughput: {npz_results['throughput']:.2f} samples/sec")
        print("\nWebDataset: SKIPPED (not available)")

    elif wd_results:
        print("\nNPZ: SKIPPED (failed)")
        print("\nWebDataset Results:")
        print(f"  Avg time: {wd_results['avg_sample_time']*1000:.2f} ms/sample")
        print(f"  Throughput: {wd_results['throughput']:.2f} samples/sec")

    else:
        print("\n❌ Both tests failed!")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
