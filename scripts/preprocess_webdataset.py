#!/usr/bin/env python3
"""
Preprocess WebDataset: Apply all fixed transformations and save to new WebDataset.

This script reads the current WebDataset (raw CT volumes with varying sizes),
applies all preprocessing steps, and saves to a new WebDataset with:
- Uniform shape: (480, 480, 240) - (H, W, D)
- Dtype: float16 (space-efficient)
- Range: [-1, 1] (normalized)

After preprocessing, DataLoader only needs to:
1. Read data (~50ms)
2. Permute dimensions (~0.01ms)
3. Add channel dimension (~0.001ms)

Total loading time: ~50-100ms (vs current ~1200ms)

Usage:
    # Process training set
    python scripts/preprocess_webdataset.py \
        --input-pattern "/path/to/train_fixed_webdataset/shard-{000000..000314}.tar" \
        --output-dir "/path/to/train_preprocessed_webdataset" \
        --num-workers 32

    # Process validation set
    python scripts/preprocess_webdataset.py \
        --input-pattern "/path/to/valid_fixed_webdataset/shard-{000000..000059}.tar" \
        --output-dir "/path/to/valid_preprocessed_webdataset" \
        --num-workers 16
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Tuple
import multiprocessing as mp
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds
from tqdm import tqdm


def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing using trilinear interpolation.

    Args:
        array (torch.Tensor): Input array to be resized, shape (1, 1, D, H, W).
        current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
        target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
        np.ndarray: Resized array, shape (D, H, W).
    """
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array


def preprocess_volume(volume_data: np.ndarray, metadata: dict) -> np.ndarray:
    """
    Apply all preprocessing steps to raw volume data.

    This replicates the logic from CTReportWebDataset._process_volume(),
    but saves the result BEFORE permute and unsqueeze (which will be done in DataLoader).

    Args:
        volume_data (np.ndarray): Raw volume data (float16, varying shape)
        metadata (dict): Metadata including spacing, rescale parameters

    Returns:
        np.ndarray: Preprocessed volume (480, 480, 240) float16, range [-1, 1]
    """
    # 1. Convert float16 to float32 for processing
    img_data = volume_data.astype(np.float32)

    # 2. Get metadata
    slope = float(metadata["RescaleSlope"])
    intercept = float(metadata["RescaleIntercept"])

    # Parse XYSpacing (format: "[0.75, 0.75]")
    xy_spacing_str = str(metadata["XYSpacing"])
    xy_spacing = float(xy_spacing_str.strip("[]").split(",")[0])
    z_spacing = float(metadata["ZSpacing"])

    # Define target spacing
    target_x_spacing = 0.75
    target_y_spacing = 0.75
    target_z_spacing = 1.5

    current_spacing = (z_spacing, xy_spacing, xy_spacing)
    target_spacing = (target_z_spacing, target_x_spacing, target_y_spacing)

    # 3. Apply rescale slope and intercept
    img_data = slope * img_data + intercept

    # 4. Clip to HU range
    hu_min, hu_max = -1000, 1000
    img_data = np.clip(img_data, hu_min, hu_max)

    # 5. Transpose to (D, H, W)
    img_data = img_data.transpose(2, 0, 1)

    # 6. Convert to tensor and add batch/channel dims
    tensor = torch.tensor(img_data, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, D, H, W)

    # 7. Resize to target spacing
    img_data = resize_array(tensor, current_spacing, target_spacing)
    img_data = img_data[0][0]  # Remove batch and channel dims -> (D, H, W)
    img_data = np.transpose(img_data, (1, 2, 0))  # (H, W, D)

    # 8. Normalize to [-1, 1] range
    img_data = (img_data / 1000).astype(np.float32)

    tensor = torch.tensor(img_data)

    # 9. Crop/pad to target shape (480, 480, 240)
    target_shape = (480, 480, 240)
    h, w, d = tensor.shape
    dh, dw, dd = target_shape

    # Calculate crop/pad indices
    h_start = max((h - dh) // 2, 0)
    h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0)
    w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0)
    d_end = min(d_start + dd, d)

    # Crop
    tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    # Pad if necessary
    pad_h_before = (dh - tensor.size(0)) // 2
    pad_h_after = dh - tensor.size(0) - pad_h_before
    pad_w_before = (dw - tensor.size(1)) // 2
    pad_w_after = dw - tensor.size(1) - pad_w_before
    pad_d_before = (dd - tensor.size(2)) // 2
    pad_d_after = dd - tensor.size(2) - pad_d_before

    tensor = torch.nn.functional.pad(
        tensor,
        (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after),
        value=-1
    )

    # 10. Convert to numpy and float16 for storage
    # Shape: (480, 480, 240)
    # NOTE: We do NOT do permute here - that will be done in DataLoader
    preprocessed = tensor.numpy().astype(np.float16)

    return preprocessed


def process_single_shard(
    shard_path: str,
    output_dir: Path,
    shard_index: int
) -> Dict:
    """
    Process a single shard: read, preprocess all samples, write to new shard.

    Args:
        shard_path: Path to input shard TAR file
        output_dir: Output directory for preprocessed shard
        shard_index: Shard index number

    Returns:
        Dict with statistics (num_samples, num_errors, shard_size, etc.)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output shard path
    shard_name = f"shard-{shard_index:06d}.tar"
    output_path = output_dir / shard_name
    temp_path = output_dir / f"{shard_name}.tmp"

    # Skip if already processed
    if output_path.exists():
        return {
            'shard_index': shard_index,
            'num_samples': 0,
            'num_errors': 0,
            'status': 'skipped',
            'message': 'Already processed'
        }

    num_samples = 0
    num_errors = 0
    error_messages = []

    try:
        # Read input shard
        dataset = wds.WebDataset(shard_path, shardshuffle=False)

        # Create output shard writer
        with wds.TarWriter(str(temp_path)) as sink:
            for sample in dataset:
                try:
                    # Decode metadata
                    metadata = json.loads(sample['json'].decode('utf-8'))
                    study_id = metadata['study_id']

                    # Decode volume
                    volume_shape = tuple(metadata['volume_shape'])
                    volume_dtype = np.dtype(metadata['volume_dtype'])
                    volume_data = np.frombuffer(sample['bin'], dtype=volume_dtype).reshape(volume_shape)

                    # Preprocess volume
                    preprocessed_volume = preprocess_volume(volume_data, metadata)

                    # Create new metadata (simplified - only keep study_id)
                    new_metadata = {
                        'study_id': study_id,
                        'volume_shape': [480, 480, 240],
                        'volume_dtype': 'float16'
                    }

                    # Write preprocessed sample
                    sink.write({
                        '__key__': sample['__key__'],
                        'bin': preprocessed_volume.tobytes(),
                        'json': json.dumps(new_metadata).encode('utf-8'),
                        'txt': sample['txt'],  # Keep original report
                        'labels': sample['labels']  # Keep original labels
                    })

                    num_samples += 1

                except Exception as e:
                    num_errors += 1
                    error_msg = f"Error processing sample {num_samples}: {e}"
                    error_messages.append(error_msg)
                    if num_errors <= 3:  # Only keep first 3 errors
                        print(f"  [Shard {shard_index}] {error_msg}")

        # Move temp file to final location
        shutil.move(str(temp_path), str(output_path))

        return {
            'shard_index': shard_index,
            'num_samples': num_samples,
            'num_errors': num_errors,
            'status': 'success',
            'shard_size_mb': output_path.stat().st_size / 1024**2,
            'errors': error_messages[:3]  # Only keep first 3
        }

    except Exception as e:
        # Clean up temp file if exists
        if temp_path.exists():
            temp_path.unlink()

        return {
            'shard_index': shard_index,
            'num_samples': num_samples,
            'num_errors': num_errors,
            'status': 'failed',
            'message': str(e)
        }


def expand_shard_pattern(pattern: str) -> list:
    """
    Expand shard pattern to list of actual file paths.

    Args:
        pattern: Shard pattern like "/path/shard-{000000..000099}.tar"

    Returns:
        List of existing shard file paths
    """
    # Extract directory and pattern
    if '{' in pattern and '}' in pattern:
        # Pattern with brace expansion
        base_dir = Path(pattern).parent
        pattern_part = Path(pattern).name

        # Extract range from pattern
        import re
        match = re.search(r'\{(\d+)\.\.(\d+)\}', pattern_part)
        if not match:
            raise ValueError(f"Invalid shard pattern: {pattern}")

        start_idx = int(match.group(1))
        end_idx = int(match.group(2))

        # Generate all shard paths
        shard_paths = []
        for i in range(start_idx, end_idx + 1):
            shard_name = re.sub(r'\{\d+\.\.\d+\}', f'{i:06d}', pattern_part)
            shard_path = base_dir / shard_name
            if shard_path.exists():
                shard_paths.append(str(shard_path))

        return shard_paths
    else:
        # Single file
        if Path(pattern).exists():
            return [pattern]
        else:
            return []


def main():
    parser = argparse.ArgumentParser(description="Preprocess WebDataset to apply fixed transformations")
    parser.add_argument('--input-pattern', type=str, required=True,
                        help='Input shard pattern (e.g., "/path/shard-{000000..000314}.tar")')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for preprocessed shards')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='Number of parallel workers (default: 16)')
    parser.add_argument('--delete-after', action='store_true',
                        help='Delete input shards after successful preprocessing (DANGEROUS!)')

    args = parser.parse_args()

    # Expand shard pattern
    print("="*80)
    print("WebDataset Preprocessing")
    print("="*80)
    print(f"Input pattern: {args.input_pattern}")
    print(f"Output directory: {args.output_dir}")
    print(f"Workers: {args.num_workers}")
    print(f"Delete after: {args.delete_after}")
    print()

    shard_paths = expand_shard_pattern(args.input_pattern)
    if not shard_paths:
        print(f"ERROR: No shards found matching pattern: {args.input_pattern}")
        return 1

    print(f"Found {len(shard_paths)} shards to process")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process shards in parallel
    print("Processing shards...")
    process_func = partial(
        process_single_shard,
        output_dir=output_dir
    )

    # Extract shard indices
    shard_data = []
    for path in shard_paths:
        import re
        match = re.search(r'shard-(\d+)\.tar', path)
        if match:
            shard_index = int(match.group(1))
            shard_data.append((path, shard_index))

    # Process with multiprocessing
    total_samples = 0
    total_errors = 0
    failed_shards = []

    with mp.Pool(processes=args.num_workers) as pool:
        results = []
        for shard_path, shard_idx in shard_data:
            result = pool.apply_async(process_single_shard, (shard_path, output_dir, shard_idx))
            results.append((shard_idx, result))

        # Progress bar
        with tqdm(total=len(results), desc="Processing shards") as pbar:
            for shard_idx, result in results:
                stats = result.get()
                total_samples += stats['num_samples']
                total_errors += stats['num_errors']

                if stats['status'] == 'failed':
                    failed_shards.append((shard_idx, stats.get('message', 'Unknown error')))

                pbar.update(1)
                pbar.set_postfix({
                    'samples': total_samples,
                    'errors': total_errors
                })

    # Summary
    print("\n" + "="*80)
    print("Preprocessing Complete!")
    print("="*80)
    print(f"Total samples processed: {total_samples}")
    print(f"Total errors: {total_errors}")
    print(f"Failed shards: {len(failed_shards)}")

    if failed_shards:
        print("\nFailed shards:")
        for shard_idx, error in failed_shards[:10]:
            print(f"  Shard {shard_idx:06d}: {error}")

    # Save manifest
    manifest = {
        'total_samples': total_samples,
        'total_errors': total_errors,
        'num_shards': len(shard_paths),
        'failed_shards': len(failed_shards),
        'volume_shape': [480, 480, 240],
        'volume_dtype': 'float16',
        'volume_range': '[-1, 1]'
    }

    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved to: {manifest_path}")

    # Delete input shards if requested
    if args.delete_after:
        print("\n⚠️  Deleting input shards...")
        for shard_path, _ in shard_data:
            try:
                Path(shard_path).unlink()
                print(f"  Deleted: {shard_path}")
            except Exception as e:
                print(f"  ERROR deleting {shard_path}: {e}")

    print("="*80)
    return 0


if __name__ == "__main__":
    exit(main())
