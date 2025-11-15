#!/usr/bin/env python3
"""
Build preprocessed WebDataset directly from Hugging Face.

This script:
1. Lists all .nii.gz files on HF (dataset/train_fixed/*.nii.gz, dataset/valid_fixed/*.nii.gz)
2. Downloads CSV files (metadata, labels, reports) for the split
3. Groups .nii.gz files into shards (e.g., 128 samples per shard)
4. Checks which shards are missing in local preprocessed webdataset
5. For missing shards:
   - Downloads the .nii.gz files
   - Reads with nibabel
   - Gets metadata/labels/reports from CSVs
   - Processes volumes (rescale, clip, resize, normalize, crop/pad)
   - Saves as WebDataset tar
   - Deletes downloaded files
6. Generates manifest.json for each split

Usage:
    # Process validation set
    python scripts/build_preprocessed_dataset.py \
        --split valid \
        --output-dir /path/to/valid_preprocessed_webdataset \
        --samples-per-shard 128 \
        --num-workers 8

    # Process training set
    python scripts/build_preprocessed_dataset.py \
        --split train \
        --output-dir /path/to/train_preprocessed_webdataset \
        --samples-per-shard 128 \
        --num-workers 16
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F
import webdataset as wds
from tqdm import tqdm
from huggingface_hub import list_repo_files, hf_hub_download


def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing.

    Args:
        array (torch.Tensor): Input array to be resized.
        current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
        target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
        torch.Tensor: Resized array.
    """
    # Add batch and channel dimensions if needed
    if array.ndim == 3:
        array = array.unsqueeze(0).unsqueeze(0)  # (D, H, W) -> (1, 1, D, H, W)
        squeeze_output = True
    else:
        squeeze_output = False

    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False)

    if squeeze_output:
        resized_array = resized_array.squeeze(0).squeeze(0)  # (1, 1, D, H, W) -> (D, H, W)

    return resized_array


def get_hf_file_list(repo_id: str, split: str) -> List[str]:
    """
    Get list of nii.gz files from Hugging Face for a specific split.

    Args:
        repo_id: HuggingFace repo ID
        split: 'train' or 'valid'

    Returns:
        List of file paths (e.g., 'dataset/train_fixed/sample_001.nii.gz')
    """
    print(f"📋 Listing files from {repo_id} (split={split})...")

    all_files = list_repo_files(repo_id=repo_id, repo_type="dataset")

    # Filter by split - look for .nii.gz files
    prefix = f"dataset/{split}_fixed/"
    split_files = [f for f in all_files if f.startswith(prefix) and f.endswith('.nii.gz')]

    print(f"   Found {len(split_files)} {split} files")

    return sorted(split_files)


def load_csv_files(repo_id: str, split: str, temp_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Download and load CSV files (metadata, labels, reports) from Hugging Face.

    Args:
        repo_id: HuggingFace repo ID
        split: 'train' or 'valid'
        temp_dir: Temporary directory for downloads

    Returns:
        Tuple of (metadata_df, labels_df, reports_df)
    """
    print(f"📥 Downloading CSV files for {split} split...")

    # Define CSV file paths on HuggingFace
    # Note: validation uses 'validation_' prefix for metadata/reports but 'valid_' for labels
    split_name = 'validation' if split == 'valid' else split
    csv_files = {
        'metadata': f'dataset/metadata/{split_name}_metadata.csv',
        'labels': f'dataset/multi_abnormality_labels/{split}_predicted_labels.csv',
        'reports': f'dataset/radiology_text_reports/{split_name}_reports.csv'
    }

    dfs = {}
    for name, file_path in csv_files.items():
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                repo_type="dataset",
                local_dir=temp_dir,
                local_dir_use_symlinks=False
            )
            df = pd.read_csv(local_path)

            # Set index to VolumeName (without .nii.gz extension)
            if 'VolumeName' in df.columns:
                df['study_id'] = df['VolumeName'].str.replace('.nii.gz', '')
                df = df.set_index('study_id')
                # Drop the original VolumeName column to avoid type conversion issues
                df = df.drop('VolumeName', axis=1, errors='ignore')

            dfs[name] = df
            print(f"   ✅ Loaded {name}: {len(df)} records")
        except Exception as e:
            print(f"   ❌ Failed to load {name}: {e}")
            raise

    return dfs['metadata'], dfs['labels'], dfs['reports']


def group_files_into_shards(files: List[str], samples_per_shard: int) -> List[List[str]]:
    """
    Group files into shards.

    Args:
        files: List of file paths
        samples_per_shard: Number of samples per shard

    Returns:
        List of shards, where each shard is a list of file paths
    """
    shards = []
    for i in range(0, len(files), samples_per_shard):
        shard_files = files[i:i + samples_per_shard]
        shards.append(shard_files)

    print(f"📦 Grouped {len(files)} files into {len(shards)} shards ({samples_per_shard} samples/shard)")

    return shards


def check_existing_shards(output_dir: Path, num_shards: int) -> List[int]:
    """
    Check which shards already exist.

    Args:
        output_dir: Output directory
        num_shards: Total number of shards

    Returns:
        List of missing shard indices
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for i in range(num_shards):
        shard_path = output_dir / f"shard-{i:06d}.tar"
        if not shard_path.exists():
            missing.append(i)

    print(f"✅ Found {num_shards - len(missing)}/{num_shards} existing shards")
    print(f"⚠️  Missing {len(missing)} shards")

    return missing


def download_files(repo_id: str, files: List[str], temp_dir: Path) -> Dict[str, Path]:
    """
    Download files from HuggingFace to temporary directory.

    Args:
        repo_id: HuggingFace repo ID
        files: List of file paths to download
        temp_dir: Temporary directory

    Returns:
        Dict mapping filename to local path
    """
    local_paths = {}

    for file_path in files:
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                repo_type="dataset",
                local_dir=temp_dir,
                local_dir_use_symlinks=False
            )

            # Extract just the filename (e.g., 'sample_001.npz')
            filename = os.path.basename(file_path)
            local_paths[filename] = Path(local_path)

        except Exception as e:
            print(f"\n❌ Failed to download {file_path}: {e}")
            raise

    return local_paths


def preprocess_volume(volume_data: np.ndarray, metadata: dict) -> np.ndarray:
    """
    Apply all preprocessing steps to raw volume data.

    This replicates the processing in webdataset_loader.py's _process_volume():
    1. Convert to float32
    2. Apply rescale slope/intercept
    3. Clip to [-1000, 1000] HU
    4. Transpose to (D, H, W)
    5. Resize to target spacing
    6. Normalize to [-1, 1] range
    7. Crop/pad to (480, 480, 240)
    8. Convert to float16
    9. Transpose back to (H, W, D) for storage

    NOTE: Permute and unsqueeze are NOT done here - they will be done in DataLoader

    Args:
        volume_data: Raw volume data (any shape, float16)
        metadata: Dict with 'spacing', 'RescaleSlope', 'RescaleIntercept'

    Returns:
        Preprocessed volume (480, 480, 240) as float16, in (H, W, D) format
    """
    # 1. Convert float16 to float32 for processing
    img_data = volume_data.astype(np.float32)

    # 2. Apply rescale slope and intercept
    slope = metadata.get('RescaleSlope', 1.0)
    intercept = metadata.get('RescaleIntercept', 0.0)
    img_data = slope * img_data + intercept

    # 3. Clip to valid HU range
    img_data = np.clip(img_data, -1000, 1000)

    # 4. Transpose from (H, W, D) to (D, H, W)
    img_data = img_data.transpose(2, 0, 1)  # (D, H, W)

    # 5. Resize to target spacing (1.5, 1.5, 1.5)
    current_spacing = metadata['spacing']
    target_spacing = [1.5, 1.5, 1.5]

    tensor = torch.from_numpy(img_data).float()
    tensor = resize_array(
        tensor,
        current_spacing=current_spacing,
        target_spacing=target_spacing
    )

    # 6. Normalize to [-1, 1] range
    tensor = tensor / 1000.0  # [-1, 1]

    # 7. Crop or pad to target shape (480, 480, 240) in (D, H, W)
    target_shape = (480, 480, 240)  # (D, H, W)

    # Get current shape
    D, H, W = tensor.shape
    tD, tH, tW = target_shape

    # Crop if needed
    if D > tD:
        start = (D - tD) // 2
        tensor = tensor[start:start + tD, :, :]
    if H > tH:
        start = (H - tH) // 2
        tensor = tensor[:, start:start + tH, :]
    if W > tW:
        start = (W - tW) // 2
        tensor = tensor[:, :, start:start + tW]

    # Pad if needed
    D, H, W = tensor.shape
    pad_d = (tD - D) // 2
    pad_h = (tH - H) // 2
    pad_w = (tW - W) // 2

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        tensor = torch.nn.functional.pad(
            tensor,
            (pad_w, tW - W - pad_w, pad_h, tH - H - pad_h, pad_d, tD - D - pad_d),
            mode='constant',
            value=-1.0
        )

    # 8. Transpose back to (H, W, D) for storage
    # WebDataset will store as (H, W, D), and loader will permute to (D, H, W) at runtime
    tensor = tensor.permute(1, 2, 0)  # (D, H, W) -> (H, W, D)

    # 9. Convert to float16 for storage efficiency
    result = tensor.numpy().astype(np.float16)

    return result  # Shape: (480, 480, 240) as float16


def process_shard(
    shard_idx: int,
    shard_files: List[str],
    repo_id: str,
    output_dir: Path,
    temp_dir: Path,
    metadata_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    reports_df: pd.DataFrame
) -> Dict:
    """
    Process one shard: download → process → save → delete.

    Args:
        shard_idx: Shard index
        shard_files: List of HF file paths for this shard
        repo_id: HuggingFace repo ID
        output_dir: Output directory for WebDataset
        temp_dir: Temporary download directory
        metadata_df: Metadata DataFrame (indexed by study_id)
        labels_df: Labels DataFrame (indexed by study_id)
        reports_df: Reports DataFrame (indexed by study_id)

    Returns:
        Dict with processing stats
    """
    shard_name = f"shard-{shard_idx:06d}"
    output_path = output_dir / f"{shard_name}.tar"

    # Create temporary subdirectory for this shard
    shard_temp_dir = temp_dir / shard_name
    shard_temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Download files
        local_paths = download_files(repo_id, shard_files, shard_temp_dir)

        # 2. Process and write to WebDataset
        with wds.TarWriter(str(output_path)) as sink:
            for filename in sorted(local_paths.keys()):
                local_path = local_paths[filename]

                # Generate study ID from filename (e.g., 'sample_001.nii.gz' -> 'sample_001')
                study_id = filename.replace('.nii.gz', '')

                # Load nii.gz file with nibabel
                nii_img = nib.load(str(local_path))
                volume_data = nii_img.get_fdata().astype(np.float32)  # (H, W, D)

                # Get spacing from NIfTI header
                header = nii_img.header
                pixdim = header['pixdim'][1:4]  # [x_spacing, y_spacing, z_spacing]
                xy_spacing = float(pixdim[0])  # Assume x and y are the same
                z_spacing = float(pixdim[2])

                # Get metadata from CSV
                if study_id not in metadata_df.index:
                    print(f"   ⚠️  Skipping {study_id}: not found in metadata")
                    continue

                meta_row = metadata_df.loc[study_id]
                slope = float(meta_row.get('RescaleSlope', 1.0))
                intercept = float(meta_row.get('RescaleIntercept', 0.0))

                # Get labels from CSV
                if study_id not in labels_df.index:
                    print(f"   ⚠️  Skipping {study_id}: not found in labels")
                    continue

                label_row = labels_df.loc[study_id]
                # Extract disease labels (assuming columns after 'VolumeName')
                disease_labels = label_row.values.astype(np.float32)

                # Get report from CSV
                if study_id not in reports_df.index:
                    print(f"   ⚠️  Skipping {study_id}: not found in reports")
                    continue

                report_row = reports_df.loc[study_id]
                findings = str(report_row.get('Findings_EN', ''))
                impressions = str(report_row.get('Impressions_EN', ''))
                report_text = findings + ' ' + impressions

                # Prepare metadata for preprocessing
                metadata = {
                    'spacing': [z_spacing, xy_spacing, xy_spacing],
                    'RescaleSlope': slope,
                    'RescaleIntercept': intercept
                }

                # Preprocess volume
                preprocessed_volume = preprocess_volume(volume_data, metadata)

                # Write to WebDataset
                sample = {
                    '__key__': study_id,
                    'bin': preprocessed_volume.tobytes(),  # (480, 480, 240) float16
                    'txt': report_text,
                    'cls': disease_labels.tobytes(),
                    'json': json.dumps({
                        'study_id': study_id,
                        'shape': list(preprocessed_volume.shape),  # [480, 480, 240]
                        'dtype': 'float16',
                        'num_classes': len(disease_labels)
                    })
                }
                sink.write(sample)

        # 3. Delete downloaded files
        shutil.rmtree(shard_temp_dir)

        return {
            'shard_idx': shard_idx,
            'num_samples': len(local_paths),
            'success': True
        }

    except Exception as e:
        # Clean up on error
        if shard_temp_dir.exists():
            shutil.rmtree(shard_temp_dir)
        if output_path.exists():
            output_path.unlink()

        return {
            'shard_idx': shard_idx,
            'num_samples': 0,
            'success': False,
            'error': str(e)
        }


def generate_manifest(output_dir: Path, split: str, num_shards: int) -> Path:
    """
    Generate manifest.json for the dataset.

    Args:
        output_dir: Output directory
        split: 'train' or 'valid'
        num_shards: Total number of shards

    Returns:
        Path to manifest.json
    """
    manifest_path = output_dir / "manifest.json"

    # Count total samples
    total_samples = 0
    shard_info = []

    for i in range(num_shards):
        shard_path = output_dir / f"shard-{i:06d}.tar"
        if shard_path.exists():
            # Count samples in this shard
            num_samples = 0
            try:
                dataset = wds.WebDataset(str(shard_path))
                for _ in dataset:
                    num_samples += 1
            except:
                num_samples = 0

            total_samples += num_samples
            shard_info.append({
                'shard_index': i,
                'filename': f"shard-{i:06d}.tar",
                'num_samples': num_samples,
                'size_bytes': shard_path.stat().st_size
            })

    # Generate manifest
    manifest = {
        'dataset': 'CT-RATE',
        'split': split,
        'format': 'webdataset',
        'preprocessed': True,
        'total_shards': num_shards,
        'total_samples': total_samples,
        'sample_shape': [480, 480, 240],
        'sample_dtype': 'float16',
        'num_classes': 18,
        'shards': shard_info
    }

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n📄 Generated manifest: {manifest_path}")
    print(f"   Total samples: {total_samples}")
    print(f"   Total shards: {num_shards}")

    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Build preprocessed WebDataset from Hugging Face"
    )
    parser.add_argument('--split', type=str, required=True, choices=['train', 'valid'],
                        help='Dataset split to process')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for preprocessed WebDataset')
    parser.add_argument('--repo-id', type=str, default='ibrahimhamamci/CT-RATE',
                        help='HuggingFace repo ID (default: ibrahimhamamci/CT-RATE)')
    parser.add_argument('--samples-per-shard', type=int, default=128,
                        help='Number of samples per shard (default: 128)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing of existing shards')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("="*80)
    print("Build Preprocessed WebDataset from Hugging Face")
    print("="*80)
    print(f"Repo ID: {args.repo_id}")
    print(f"Split: {args.split}")
    print(f"Output dir: {output_dir}")
    print(f"Samples per shard: {args.samples_per_shard}")
    print(f"Workers: {args.num_workers}")
    print()

    # 1. Get file list from HuggingFace
    hf_files = get_hf_file_list(args.repo_id, args.split)

    if len(hf_files) == 0:
        print(f"❌ No files found for split '{args.split}'")
        return 1

    # 2. Group into shards
    shards = group_files_into_shards(hf_files, args.samples_per_shard)

    # 3. Check existing shards
    if args.force:
        missing_shards = list(range(len(shards)))
        print(f"⚡ Force mode: Processing all {len(shards)} shards")
    else:
        missing_shards = check_existing_shards(output_dir, len(shards))

    if len(missing_shards) == 0:
        print("\n✅ All shards already exist! Use --force to reprocess.")
        # Generate manifest even if no missing shards
        generate_manifest(output_dir, args.split, len(shards))
        return 0

    # 4. Load CSV files (metadata, labels, reports)
    print(f"\n📚 Loading CSV files...")
    with tempfile.TemporaryDirectory(prefix=f"ct_rate_csv_") as csv_temp_dir:
        csv_temp_path = Path(csv_temp_dir)
        metadata_df, labels_df, reports_df = load_csv_files(
            args.repo_id,
            args.split,
            csv_temp_path
        )

    print(f"   Loaded metadata: {len(metadata_df)} records")
    print(f"   Loaded labels: {len(labels_df)} records")
    print(f"   Loaded reports: {len(reports_df)} records")

    # 5. Process missing shards
    print(f"\n🔄 Processing {len(missing_shards)} missing shards...")

    # Create temporary directory
    with tempfile.TemporaryDirectory(prefix=f"ct_rate_{args.split}_") as temp_dir:
        temp_path = Path(temp_dir)

        # Process shards in parallel
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {}
            for shard_idx in missing_shards:
                future = executor.submit(
                    process_shard,
                    shard_idx,
                    shards[shard_idx],
                    args.repo_id,
                    output_dir,
                    temp_path,
                    metadata_df,
                    labels_df,
                    reports_df
                )
                futures[future] = shard_idx

            # Track progress
            pbar = tqdm(total=len(missing_shards), desc="Processing shards", unit="shard")
            success_count = 0
            fail_count = 0

            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)

                if result['success']:
                    success_count += 1
                    pbar.set_postfix({
                        'success': success_count,
                        'failed': fail_count
                    })
                else:
                    fail_count += 1
                    print(f"\n❌ Shard {result['shard_idx']:06d} failed: {result.get('error', 'Unknown error')}")
                    pbar.set_postfix({
                        'success': success_count,
                        'failed': fail_count
                    })

            pbar.close()

    # 6. Generate manifest
    generate_manifest(output_dir, args.split, len(shards))

    # 7. Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"✅ Success: {success_count}/{len(missing_shards)} shards")
    print(f"❌ Failed: {fail_count}/{len(missing_shards)} shards")
    print(f"📁 Output: {output_dir}")
    print("="*80)

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    exit(main())
