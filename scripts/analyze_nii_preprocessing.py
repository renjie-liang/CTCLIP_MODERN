#!/usr/bin/env python3
"""
Analyze nii.gz files to infer what preprocessing has been done.

Usage:
    python scripts/analyze_nii_preprocessing.py \
        --data-dir /path/to/valid_fixed \
        --num-samples 100
"""

import argparse
import glob
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
import nibabel as nib
from tqdm import tqdm


def analyze_single_file(nii_path, base_dir=None):
    """Analyze a single nii.gz file and extract statistics."""
    try:
        # Load NIfTI file
        nii_img = nib.load(str(nii_path))

        # Get data array (header access is faster than get_fdata if we don't need pixel values)
        # But we need min/max so we have to load data
        data = nii_img.get_fdata()

        # Get header information
        pixdim = nii_img.header.get_zooms() # [x_spacing, y_spacing, z_spacing]
        
        # --- NEW: Get Orientation ---
        # aff2axcodes return tuple like ('R', 'A', 'S') or ('L', 'P', 'S')
        orientation = nib.aff2axcodes(nii_img.affine)
        orientation_str = "".join(orientation) # e.g., "RAS", "LPS"

        # Extract statistics
        stats = {
            'file': Path(nii_path).name,
            'relative_path': str(Path(nii_path).relative_to(base_dir)) if base_dir else Path(nii_path).name,
            'shape': data.shape,
            'dtype': str(data.dtype),
            'spacing': tuple(pixdim),
            'orientation': orientation_str,  # <--- Added
            'min_value': float(np.min(data)),
            'max_value': float(np.max(data)),
            'mean_value': float(np.mean(data)),
            'std_value': float(np.std(data)),
            'unique_values_count': len(np.unique(data)),
            'has_negative': bool(np.any(data < 0)),
            'has_values_gt_1000': bool(np.any(data > 1000)),
            'has_values_in_range_minus1_to_1': bool(np.all((data >= -1.1) & (data <= 1.1))),
        }

        # Check if values look like HU units
        # HU range: air=-1000, water=0, bone=+1000 to +3000
        stats['looks_like_HU'] = (stats['min_value'] < -500 and stats['max_value'] > 500)

        # Check if normalized to [-1, 1]
        stats['looks_normalized'] = (stats['min_value'] >= -1.05 and stats['max_value'] <= 1.05)

        return stats

    except Exception as e:
        print(f"Error analyzing {nii_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Analyze nii.gz preprocessing status")
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing nii.gz files')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of files to sample')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for sampling')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("="*80)
    print("NIfTI Preprocessing Analysis")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Sample size: {args.num_samples}")
    print()

    # Find all nii.gz files (recursive search for nested directories)
    print("🔍 Searching for nii.gz files (recursive)...")
    nii_files = list(data_dir.rglob("*.nii.gz"))

    if len(nii_files) == 0:
        print(f"❌ No nii.gz files found in {data_dir}")
        print(f"   Tried recursive search with pattern: **/*.nii.gz")
        return

    print(f"   Found {len(nii_files)} total files")

    # Analyze directory structure depth
    depths = []
    for f in nii_files[:100]:  # Sample first 100 for structure analysis
        relative_path = f.relative_to(data_dir)
        depth = len(relative_path.parts) - 1  # Exclude the filename
        depths.append(depth)

    if depths:
        max_depth = max(depths)
        min_depth = min(depths)
        print(f"   Directory depth: {min_depth} to {max_depth} levels")
        print(f"   Example path: {nii_files[0].relative_to(data_dir)}")

    # Sample files
    random.seed(args.random_seed)
    if len(nii_files) > args.num_samples:
        sampled_files = random.sample(nii_files, args.num_samples)
        print(f"   Randomly sampling {args.num_samples} files")
    else:
        sampled_files = nii_files
        print(f"   Using all {len(nii_files)} files")

    # Analyze files
    print(f"\n📊 Analyzing {len(sampled_files)} files...")
    all_stats = []

    for nii_path in tqdm(sampled_files, desc="Analyzing", unit="file"):
        stats = analyze_single_file(nii_path, base_dir=data_dir)
        if stats:
            all_stats.append(stats)

    print(f"\n✅ Successfully analyzed {len(all_stats)} files")

    # Aggregate statistics
    print("\n" + "="*80)
    print("AGGREGATE STATISTICS")
    print("="*80)

    # --- NEW: Orientation Distribution ---
    print("\n🧭 Orientation Distribution:")
    orient_counts = defaultdict(int)
    for s in all_stats:
        orient_counts[s['orientation']] += 1
    
    for orient, count in sorted(orient_counts.items(), key=lambda x: -x[1]):
        percentage = (count / len(all_stats)) * 100
        print(f"   {orient}: {count} files ({percentage:.1f}%)")

    # Shape distribution
    print("\n📐 Shape Distribution:")
    shape_counts = defaultdict(int)
    for s in all_stats:
        shape_counts[s['shape']] += 1

    for shape, count in sorted(shape_counts.items(), key=lambda x: -x[1]):
        percentage = (count / len(all_stats)) * 100
        print(f"   {shape}: {count} files ({percentage:.1f}%)")

    # Spacing distribution
    print("\n📏 Spacing Distribution:")
    spacing_counts = defaultdict(int)
    for s in all_stats:
        # Round to 2 decimal places for grouping
        rounded_spacing = tuple(round(x, 2) for x in s['spacing'])
        spacing_counts[rounded_spacing] += 1

    for spacing, count in sorted(spacing_counts.items(), key=lambda x: -x[1])[:10]:
        percentage = (count / len(all_stats)) * 100
        print(f"   {spacing}: {count} files ({percentage:.1f}%)")

    # Data type
    print("\n🔢 Data Type Distribution:")
    dtype_counts = defaultdict(int)
    for s in all_stats:
        dtype_counts[s['dtype']] += 1

    for dtype, count in sorted(dtype_counts.items(), key=lambda x: -x[1]):
        percentage = (count / len(all_stats)) * 100
        print(f"   {dtype}: {count} files ({percentage:.1f}%)")

    # Value range statistics
    print("\n📈 Value Range Statistics:")
    min_values = [s['min_value'] for s in all_stats]
    max_values = [s['max_value'] for s in all_stats]
    mean_values = [s['mean_value'] for s in all_stats]
    std_values = [s['std_value'] for s in all_stats]

    print(f"   Min value range: [{np.min(min_values):.2f}, {np.max(min_values):.2f}]")
    print(f"   Max value range: [{np.min(max_values):.2f}, {np.max(max_values):.2f}]")
    print(f"   Mean value range: [{np.min(mean_values):.2f}, {np.max(mean_values):.2f}]")
    print(f"   Std value range: [{np.min(std_values):.2f}, {np.max(std_values):.2f}]")

    # Percentage with specific characteristics
    print("\n🔍 Characteristic Analysis:")

    has_negative = sum(1 for s in all_stats if s['has_negative'])
    print(f"   Files with negative values: {has_negative}/{len(all_stats)} ({has_negative/len(all_stats)*100:.1f}%)")

    looks_like_HU = sum(1 for s in all_stats if s['looks_like_HU'])
    print(f"   Files that look like HU values: {looks_like_HU}/{len(all_stats)} ({looks_like_HU/len(all_stats)*100:.1f}%)")

    looks_normalized = sum(1 for s in all_stats if s['looks_normalized'])
    print(f"   Files that look normalized [-1,1]: {looks_normalized}/{len(all_stats)} ({looks_normalized/len(all_stats)*100:.1f}%)")

    has_values_gt_1000 = sum(1 for s in all_stats if s['has_values_gt_1000'])
    print(f"   Files with values > 1000: {has_values_gt_1000}/{len(all_stats)} ({has_values_gt_1000/len(all_stats)*100:.1f}%)")

    # Infer preprocessing status
    print("\n" + "="*80)
    print("PREPROCESSING INFERENCE")
    print("="*80)

    # Check if uniform orientation
    unique_orients = len(orient_counts)
    if unique_orients == 1:
        orient = list(orient_counts.keys())[0]
        print(f"✅ UNIFORM ORIENTATION: All files are {orient}")
    else:
        print(f"⚠️  VARIABLE ORIENTATION: {unique_orients} different orientations detected")
        print(f"   Main orientations: {dict(list(orient_counts.items())[:3])}")

    # Check if uniform spacing
    unique_spacings = len(spacing_counts)
    if unique_spacings == 1:
        spacing = list(spacing_counts.keys())[0]
        print(f"✅ UNIFORM SPACING: All files have spacing {spacing}")
    else:
        print(f"⚠️  VARIABLE SPACING: {unique_spacings} different spacings detected")

    # Check if uniform shape
    unique_shapes = len(shape_counts)
    if unique_shapes == 1:
        shape = list(shape_counts.keys())[0]
        print(f"✅ UNIFORM SHAPE: All files have shape {shape}")
    else:
        print(f"⚠️  VARIABLE SHAPE: {unique_shapes} different shapes detected")

    # Check value range
    if looks_like_HU > len(all_stats) * 0.8:
        print(f"✅ HU VALUES: {looks_like_HU/len(all_stats)*100:.1f}% files appear to be in HU units")
        print(f"   → NOT normalized yet")
    elif looks_normalized > len(all_stats) * 0.8:
        print(f"✅ NORMALIZED: {looks_normalized/len(all_stats)*100:.1f}% files appear normalized to [-1, 1]")
        print(f"   → Already normalized")
    else:
        print(f"⚠️  MIXED VALUES: Unable to determine if HU or normalized")

    # Summary of preprocessing status
    print("\n" + "="*80)
    print("SUMMARY: What preprocessing has been done?")
    print("="*80)

    preprocessing_done = []
    preprocessing_todo = []

    if unique_orients == 1:
        orient_val = list(orient_counts.keys())[0]
        preprocessing_done.append(f"✅ Standardized orientation: {orient_val}")
    else:
        preprocessing_todo.append("❌ Need to standardize orientation (e.g. to RAS)")

    if unique_spacings == 1:
        spacing_val = list(spacing_counts.keys())[0]
        preprocessing_done.append(f"✅ Resampled to uniform spacing: {spacing_val}")
    else:
        preprocessing_todo.append("❌ Need to resample to uniform spacing")

    if unique_shapes == 1:
        shape_val = list(shape_counts.keys())[0]
        preprocessing_done.append(f"✅ Cropped/padded to uniform shape: {shape_val}")
    else:
        preprocessing_todo.append("❌ Need to crop/pad to uniform shape")

    if looks_normalized > len(all_stats) * 0.8:
        preprocessing_done.append("✅ Normalized to [-1, 1] range")
    else:
        preprocessing_todo.append("❌ Need to normalize to [-1, 1]")

    print("\nPreprocessing DONE:")
    for item in preprocessing_done:
        print(f"   {item}")

    print("\nPreprocessing TODO:")
    if preprocessing_todo:
        for item in preprocessing_todo:
            print(f"   {item}")
    else:
        print("   None - fully preprocessed!")

    # Show a few example files with detailed stats
    print("\n" + "="*80)
    print("EXAMPLE FILES (first 5)")
    print("="*80)

    for i, stats in enumerate(all_stats[:5]):
        print(f"\n[{i+1}] {stats['relative_path']}")
        print(f"    Orientation: {stats['orientation']}")  # <--- Show orientation
        print(f"    Shape: {stats['shape']}")
        print(f"    Spacing: {stats['spacing']}")
        print(f"    Dtype: {stats['dtype']}")
        print(f"    Value range: [{stats['min_value']:.2f}, {stats['max_value']:.2f}]")
        print(f"    Mean: {stats['mean_value']:.2f}, Std: {stats['std_value']:.2f}")
        print(f"    Looks like HU: {stats['looks_like_HU']}")
        print(f"    Looks normalized: {stats['looks_normalized']}")

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()