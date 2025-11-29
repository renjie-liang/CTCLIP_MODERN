#!/usr/bin/env python3
"""
Script to create WebDataset subsets using symlinks.

This script reads the original manifest.json and creates subset directories
with symlinks and accurate manifest.json files.
"""

import json
import os
from pathlib import Path
import argparse


def create_subset(source_dir, subset_dir, num_shards, subset_name):
    """
    Create a subset with symlinks and accurate manifest.

    Args:
        source_dir: Source directory containing shards and manifest.json
        subset_dir: Target directory for subset
        num_shards: Number of shards to include (0 to num_shards-1)
        subset_name: Name for description (e.g., "training", "validation")
    """
    source_path = Path(source_dir)
    subset_path = Path(subset_dir)

    # Create subset directory
    subset_path.mkdir(parents=True, exist_ok=True)

    # Read original manifest
    manifest_file = source_path / 'manifest.json'
    if not manifest_file.exists():
        print(f"  ⚠️  Warning: {manifest_file} not found, using estimation")
        return create_subset_without_manifest(source_dir, subset_dir, num_shards, subset_name)

    with open(manifest_file, 'r') as f:
        original_manifest = json.load(f)

    # Calculate exact sample count from manifest
    total_samples = 0
    subset_shards = []

    for i in range(num_shards):
        shard_info = original_manifest['shards'][i]
        shard_filename = shard_info['filename']
        num_samples = shard_info['num_samples']

        # Create symlink
        source_shard = source_path / shard_filename
        target_shard = subset_path / shard_filename

        if source_shard.exists():
            # Remove existing symlink if it exists
            if target_shard.is_symlink() or target_shard.exists():
                target_shard.unlink()
            # Create new symlink
            target_shard.symlink_to(source_shard)
        else:
            print(f"  ⚠️  Warning: {source_shard} not found")
            continue

        total_samples += num_samples
        subset_shards.append({
            "shard_index": i,
            "filename": shard_filename,
            "num_samples": num_samples,
            "size_bytes": shard_info.get('size_bytes', 0)
        })

    # Create new manifest
    new_manifest = {
        "dataset": original_manifest.get('dataset', 'CT-RATE'),
        "split": original_manifest.get('split', subset_name),
        "format": "webdataset",
        "preprocessed": True,
        "total_shards": num_shards,
        "total_samples": total_samples,
        "sample_shape": original_manifest.get('sample_shape', [480, 480, 240]),
        "sample_dtype": original_manifest.get('sample_dtype', 'float16'),
        "num_classes": original_manifest.get('num_classes', 18),
        "description": f"Subset of {subset_name} data (shards 0-{num_shards-1})",
        "created_from": str(source_path),
        "shards": subset_shards
    }

    # Write manifest
    manifest_output = subset_path / 'manifest.json'
    with open(manifest_output, 'w') as f:
        json.dump(new_manifest, f, indent=2)

    return total_samples, num_shards


def create_subset_without_manifest(source_dir, subset_dir, num_shards, subset_name):
    """Fallback: create subset without manifest (use estimation)."""
    source_path = Path(source_dir)
    subset_path = Path(subset_dir)

    subset_path.mkdir(parents=True, exist_ok=True)

    # Create symlinks
    for i in range(num_shards):
        shard_filename = f"shard-{i:06d}.tar"
        source_shard = source_path / shard_filename
        target_shard = subset_path / shard_filename

        if source_shard.exists():
            if target_shard.is_symlink() or target_shard.exists():
                target_shard.unlink()
            target_shard.symlink_to(source_shard)

    # Estimate sample count (you'll need to update these based on your data)
    estimated_samples = num_shards * 150  # Rough estimate

    # Create simple manifest
    manifest = {
        "total_samples": estimated_samples,
        "num_shards": num_shards,
        "description": f"Subset of {subset_name} data (shards 0-{num_shards-1}) - ESTIMATED",
        "created_from": str(source_path)
    }

    with open(subset_path / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    return estimated_samples, num_shards


def main():
    parser = argparse.ArgumentParser(description='Create WebDataset subsets with symlinks')
    parser.add_argument('--base-dir',
                        default='/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset',
                        help='Base directory for datasets')
    parser.add_argument('--train-shards', type=int, default=50,
                        help='Number of training shards (default: 50)')
    parser.add_argument('--val-shards', type=int, default=10,
                        help='Number of validation shards (default: 10)')

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    train_source = base_dir / 'train_preprocessed_webdataset'
    val_source = base_dir / 'valid_preprocessed_webdataset'

    print("=" * 60)
    print("Creating WebDataset Subsets")
    print("=" * 60)

    # Create training subset
    print(f"\n📦 Creating training subset ({args.train_shards} shards)...")
    train_subset = base_dir / f'train_subset_{args.train_shards}'
    train_samples, train_shards = create_subset(
        train_source,
        train_subset,
        args.train_shards,
        "training"
    )
    print(f"  ✓ Training subset created: {train_subset}")
    print(f"    Shards: {train_shards}")
    print(f"    Samples: {train_samples}")

    # Create validation subset
    print(f"\n📦 Creating validation subset ({args.val_shards} shards)...")
    val_subset = base_dir / f'val_subset_{args.val_shards}'
    val_samples, val_shards = create_subset(
        val_source,
        val_subset,
        args.val_shards,
        "validation"
    )
    print(f"  ✓ Validation subset created: {val_subset}")
    print(f"    Shards: {val_shards}")
    print(f"    Samples: {val_samples}")

    # Summary
    print("\n" + "=" * 60)
    print("Subset Creation Complete!")
    print("=" * 60)
    print(f"\n📊 Training subset:")
    print(f"  Path: {train_subset}")
    print(f"  Pattern: {train_subset}/shard-{{000000..{args.train_shards-1:06d}}}.tar")
    print(f"  Samples: {train_samples}")

    print(f"\n📊 Validation subset:")
    print(f"  Path: {val_subset}")
    print(f"  Pattern: {val_subset}/shard-{{000000..{args.val_shards-1:06d}}}.tar")
    print(f"  Samples: {val_samples}")

    print(f"\n📝 Update your config with:")
    print(f"  webdataset_shards_train: {train_subset}/shard-{{000000..{args.train_shards-1:06d}}}.tar")
    print(f"  webdataset_shards_val: {val_subset}/shard-{{000000..{args.val_shards-1:06d}}}.tar")
    print()


if __name__ == '__main__':
    main()
