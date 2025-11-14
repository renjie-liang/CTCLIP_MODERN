#!/usr/bin/env python3
"""
Verify that preprocessed WebDataset produces identical results to original processing.

This script:
1. Loads samples from original WebDataset
2. Processes them using _process_volume()
3. Loads same samples from preprocessed WebDataset
4. Compares the outputs to ensure they are identical

Usage:
    python scripts/verify_preprocessed_data.py \
        --original-pattern "/path/to/train_fixed_webdataset/shard-{000000..000001}.tar" \
        --preprocessed-pattern "/path/to/train_preprocessed_webdataset/shard-{000000..000001}.tar" \
        --num-samples 100
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from src.data.webdataset_loader import CTReportWebDataset


def compare_tensors(tensor1, tensor2, name="Tensor", rtol=1e-5, atol=1e-6):
    """
    Compare two tensors and report differences.

    Args:
        tensor1: First tensor
        tensor2: Second tensor
        name: Name for logging
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        bool: True if tensors are close, False otherwise
    """
    if tensor1.shape != tensor2.shape:
        print(f"❌ {name} shape mismatch: {tensor1.shape} vs {tensor2.shape}")
        return False

    # Convert to numpy for comparison
    arr1 = tensor1.cpu().numpy() if isinstance(tensor1, torch.Tensor) else tensor1
    arr2 = tensor2.cpu().numpy() if isinstance(tensor2, torch.Tensor) else tensor2

    # Check if close
    if np.allclose(arr1, arr2, rtol=rtol, atol=atol):
        print(f"✅ {name} matches (shape={tensor1.shape})")
        return True
    else:
        # Calculate differences
        diff = np.abs(arr1 - arr2)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        num_diff = np.sum(~np.isclose(arr1, arr2, rtol=rtol, atol=atol))
        total_elements = arr1.size

        print(f"❌ {name} MISMATCH:")
        print(f"   Shape: {tensor1.shape}")
        print(f"   Max difference: {max_diff}")
        print(f"   Mean difference: {mean_diff}")
        print(f"   Different elements: {num_diff}/{total_elements} ({num_diff/total_elements*100:.2f}%)")
        print(f"   Value range [original]: [{arr1.min():.4f}, {arr1.max():.4f}]")
        print(f"   Value range [preprocessed]: [{arr2.min():.4f}, {arr2.max():.4f}]")

        return False


def verify_samples(original_dataset, preprocessed_dataset, num_samples=10):
    """
    Verify that preprocessed dataset produces identical results to original.

    Args:
        original_dataset: CTReportWebDataset with preprocessed=False
        preprocessed_dataset: CTReportWebDataset with preprocessed=True
        num_samples: Number of samples to verify

    Returns:
        Dict with verification results
    """
    print("="*80)
    print("Verifying Preprocessed Data")
    print("="*80)
    print(f"Samples to verify: {num_samples}")
    print()

    # Create loaders
    original_loader = original_dataset.create_pytorch_dataloader(
        batch_size=1,
        num_workers=0,  # Single process for deterministic order
        prefetch_factor=2
    )

    preprocessed_loader = preprocessed_dataset.create_pytorch_dataloader(
        batch_size=1,
        num_workers=0,  # Single process for deterministic order
        prefetch_factor=2
    )

    num_passed = 0
    num_failed = 0
    failures = []

    # Iterate through samples
    for i, (orig_batch, prep_batch) in enumerate(zip(original_loader, preprocessed_loader)):
        if i >= num_samples:
            break

        # Unpack batches
        orig_volume, orig_report, orig_labels, orig_id, _ = orig_batch
        prep_volume, prep_report, prep_labels, prep_id, _ = prep_batch

        # Verify study IDs match
        orig_id_str = orig_id[0] if isinstance(orig_id, (list, tuple)) else str(orig_id)
        prep_id_str = prep_id[0] if isinstance(prep_id, (list, tuple)) else str(prep_id)

        print(f"\nSample {i+1}/{num_samples}: {orig_id_str}")
        print("-" * 80)

        if orig_id_str != prep_id_str:
            print(f"❌ Study ID mismatch: {orig_id_str} vs {prep_id_str}")
            num_failed += 1
            failures.append({
                'sample_index': i,
                'original_id': orig_id_str,
                'preprocessed_id': prep_id_str,
                'error': 'Study ID mismatch'
            })
            continue

        # Remove batch dimension for comparison
        orig_volume = orig_volume[0]  # (1, D, H, W)
        prep_volume = prep_volume[0]  # (1, D, H, W)
        orig_labels = orig_labels[0]  # (num_classes,)
        prep_labels = prep_labels[0]  # (num_classes,)

        # Compare volumes
        volume_match = compare_tensors(
            orig_volume, prep_volume,
            name=f"Volume [{orig_id_str}]",
            rtol=1e-4,  # Slightly relaxed due to float16 conversion
            atol=1e-4
        )

        # Compare labels
        labels_match = compare_tensors(
            orig_labels, prep_labels,
            name=f"Labels [{orig_id_str}]",
            rtol=1e-5,
            atol=1e-6
        )

        # Compare reports (should be identical strings)
        report_match = orig_report == prep_report
        if report_match:
            print(f"✅ Report matches")
        else:
            print(f"❌ Report MISMATCH")

        # Overall result
        if volume_match and labels_match and report_match:
            num_passed += 1
            print(f"✅ Sample {i+1} PASSED")
        else:
            num_failed += 1
            failures.append({
                'sample_index': i,
                'study_id': orig_id_str,
                'volume_match': volume_match,
                'labels_match': labels_match,
                'report_match': report_match
            })
            print(f"❌ Sample {i+1} FAILED")

    # Summary
    print("\n" + "="*80)
    print("Verification Summary")
    print("="*80)
    print(f"Total samples: {num_samples}")
    print(f"Passed: {num_passed} ({num_passed/num_samples*100:.1f}%)")
    print(f"Failed: {num_failed} ({num_failed/num_samples*100:.1f}%)")

    if failures:
        print(f"\nFirst {min(5, len(failures))} failures:")
        for failure in failures[:5]:
            print(f"  Sample {failure['sample_index']}: {failure.get('study_id', 'unknown')}")
            if 'error' in failure:
                print(f"    Error: {failure['error']}")
            else:
                print(f"    Volume: {'✅' if failure.get('volume_match') else '❌'}")
                print(f"    Labels: {'✅' if failure.get('labels_match') else '❌'}")
                print(f"    Report: {'✅' if failure.get('report_match') else '❌'}")

    print("="*80)

    return {
        'num_samples': num_samples,
        'num_passed': num_passed,
        'num_failed': num_failed,
        'pass_rate': num_passed / num_samples,
        'failures': failures
    }


def main():
    parser = argparse.ArgumentParser(description="Verify preprocessed WebDataset")
    parser.add_argument('--original-pattern', type=str, required=True,
                        help='Original WebDataset pattern')
    parser.add_argument('--preprocessed-pattern', type=str, required=True,
                        help='Preprocessed WebDataset pattern')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples to verify (default: 10)')

    args = parser.parse_args()

    print("="*80)
    print("WebDataset Verification")
    print("="*80)
    print(f"Original pattern: {args.original_pattern}")
    print(f"Preprocessed pattern: {args.preprocessed_pattern}")
    print(f"Num samples: {args.num_samples}")
    print()

    # Create datasets
    print("Creating original dataset...")
    original_dataset = CTReportWebDataset(
        shard_pattern=args.original_pattern,
        shuffle=False,  # Deterministic order for verification
        buffer_size=0,
        preprocessed=False,  # Use full preprocessing
        mode="verify_original"
    )

    print("Creating preprocessed dataset...")
    preprocessed_dataset = CTReportWebDataset(
        shard_pattern=args.preprocessed_pattern,
        shuffle=False,  # Deterministic order for verification
        buffer_size=0,
        preprocessed=True,  # Use fast loading
        mode="verify_preprocessed"
    )

    # Verify
    results = verify_samples(original_dataset, preprocessed_dataset, args.num_samples)

    # Exit with appropriate code
    if results['pass_rate'] == 1.0:
        print("\n✅ All samples passed verification!")
        return 0
    else:
        print(f"\n❌ {results['num_failed']} samples failed verification")
        return 1


if __name__ == "__main__":
    exit(main())
