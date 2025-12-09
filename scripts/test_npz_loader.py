"""
Quick test script for NPZ DataLoader.

Usage:
    python scripts/test_npz_loader.py
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.npz_loader import CTReportNPZDataset


def test_single_sample():
    """Test loading a single sample."""
    print("="*80)
    print("Test 1: Single Sample Loading")
    print("="*80)

    # Create dataset
    dataset = CTReportNPZDataset(
        data_folder="/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/vaild_npz",
        reports_file="/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/radiology_text_reports/validation_reports.csv",
        meta_file="/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/metadata/validation_metadata.csv",
        labels_file="/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/multi_abnormality_labels/valid_predicted_labels.csv",
        mode="val"
    )

    print(f"\nDataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("ERROR: No samples found!")
        return False

    # Load first sample
    print("\nLoading first sample...")
    volume, report, labels, study_id = dataset[0]

    # Check shapes
    print(f"\n✓ Volume shape: {volume.shape}")
    print(f"  Expected: (1, 240, 480, 480)")
    assert volume.shape == (1, 240, 480, 480), f"Wrong shape: {volume.shape}"

    # Check data type
    print(f"✓ Volume dtype: {volume.dtype}")
    assert volume.dtype == torch.float32, f"Wrong dtype: {volume.dtype}"

    # Check value range
    vol_min = volume.min().item()
    vol_max = volume.max().item()
    print(f"✓ Volume range: [{vol_min:.3f}, {vol_max:.3f}]")
    print(f"  Expected: approximately [-1.0, 1.0]")

    # Check labels
    print(f"✓ Labels shape: {labels.shape}")
    print(f"  Num classes: {len(labels)}")

    # Check report
    print(f"✓ Report length: {len(report)} chars")
    print(f"  Study ID: {study_id}")
    print(f"  Report preview: {report[:100]}...")

    print("\n" + "="*80)
    print("Test 1: PASSED ✓")
    print("="*80)
    return True


def test_dataloader():
    """Test PyTorch DataLoader."""
    print("\n" + "="*80)
    print("Test 2: DataLoader (Batch Loading)")
    print("="*80)

    # Create dataset
    dataset = CTReportNPZDataset(
        data_folder="/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/vaild_npz",
        reports_file="/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/radiology_text_reports/validation_reports.csv",
        meta_file="/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/metadata/validation_metadata.csv",
        labels_file="/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/multi_abnormality_labels/valid_predicted_labels.csv",
        mode="val"
    )

    # Create DataLoader
    print("\nCreating DataLoader (batch_size=2, num_workers=0)...")
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print(f"DataLoader created: {len(loader)} batches")

    # Load first batch
    print("\nLoading first batch...")
    for batch in loader:
        volumes, reports, labels, study_ids = batch

        print(f"✓ Batch volumes shape: {volumes.shape}")
        print(f"  Expected: (2, 1, 240, 480, 480)")
        assert volumes.shape == (2, 1, 240, 480, 480), f"Wrong batch shape: {volumes.shape}"

        print(f"✓ Batch labels shape: {labels.shape}")
        print(f"✓ Number of reports: {len(reports)}")
        print(f"✓ Number of study IDs: {len(study_ids)}")

        break  # Only test first batch

    print("\n" + "="*80)
    print("Test 2: PASSED ✓")
    print("="*80)
    return True


def test_timing():
    """Test loading speed for a few samples."""
    import time

    print("\n" + "="*80)
    print("Test 3: Loading Speed (10 samples)")
    print("="*80)

    # Create dataset
    dataset = CTReportNPZDataset(
        data_folder="/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/vaild_npz",
        reports_file="/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/radiology_text_reports/validation_reports.csv",
        meta_file="/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/metadata/validation_metadata.csv",
        labels_file="/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/multi_abnormality_labels/valid_predicted_labels.csv",
        mode="val"
    )

    num_samples = min(10, len(dataset))
    times = []

    print(f"\nLoading {num_samples} samples...")
    for i in range(num_samples):
        start = time.time()
        _ = dataset[i]
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    print(f"\n✓ Average loading time: {avg_time*1000:.2f} ms/sample")
    print(f"✓ Throughput: {1/avg_time:.2f} samples/sec")

    print("\n" + "="*80)
    print("Test 3: PASSED ✓")
    print("="*80)
    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("NPZ DataLoader Test Suite")
    print("="*80)

    try:
        # Test 1: Single sample
        if not test_single_sample():
            print("\n❌ Test 1 failed!")
            return

        # Test 2: DataLoader
        if not test_dataloader():
            print("\n❌ Test 2 failed!")
            return

        # Test 3: Timing
        if not test_timing():
            print("\n❌ Test 3 failed!")
            return

        # All tests passed
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        print("\nNPZ DataLoader is ready to use! 🚀")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
