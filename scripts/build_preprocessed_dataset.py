#!/usr/bin/env python3
"""
Build preprocessed WebDataset directly from Hugging Face (Simplified version).

This script:
1. Lists all .nii.gz files on HF
2. Scans existing shards to find what's already processed
3. Processes only missing files
4. Supports appending to the last incomplete shard
5. Single-threaded, simple, easy to debug

Usage:
    python scripts/build_preprocessed_dataset.py \
        --split train \
        --output-dir /path/to/output \
        --repo-id ibrahimhamamci/CT-RATE \
        --temp-dir /path/to/temp
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

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


@contextmanager
def managed_temp_dir(base_dir: Optional[Path], prefix: str):
    """
    Context manager for temporary directory.
    If base_dir is provided, creates a subdirectory there and cleans up after.
    Otherwise, uses system tempfile.TemporaryDirectory.
    """
    if base_dir:
        # User-specified temp directory
        temp_path = base_dir / f"{prefix}_{os.getpid()}"
        temp_path.mkdir(parents=True, exist_ok=True)
        try:
            yield str(temp_path)
        finally:
            # Cleanup
            if temp_path.exists():
                shutil.rmtree(temp_path)
    else:
        # System temp directory
        with tempfile.TemporaryDirectory(prefix=prefix) as temp_dir:
            yield temp_dir


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

    print(f"   Found {len(split_files)} {split} files on HuggingFace")

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
    split_name = 'validation' if split == 'valid' else split
    csv_files = {
        'metadata': f'dataset/metadata/{split_name}_metadata.csv',
        'labels': f'dataset/multi_abnormality_labels/{split}_predicted_labels.csv',
        'reports': f'dataset/radiology_text_reports/{split_name}_reports.csv'
    }

    dfs = {}
    for name, file_path in csv_files.items():
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
            df = df.drop('VolumeName', axis=1, errors='ignore')

        dfs[name] = df
        print(f"   ✅ Loaded {name}: {len(df)} records")

    return dfs['metadata'], dfs['labels'], dfs['reports']


def preprocess_volume(volume_data: np.ndarray, metadata: dict) -> np.ndarray:
    """
    Apply all preprocessing steps to raw volume data.

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
    tensor = tensor.permute(1, 2, 0)  # (D, H, W) -> (H, W, D)

    # 9. Convert to float16 for storage efficiency
    result = tensor.numpy().astype(np.float16)

    return result  # Shape: (480, 480, 240) as float16


def get_existing_samples(output_dir: Path) -> Set[str]:
    """
    Scan all existing shards and return set of already processed study_ids.

    Args:
        output_dir: Directory containing shard-*.tar files

    Returns:
        Set of study_ids that have been processed
    """
    existing = set()

    if not output_dir.exists():
        return existing

    shard_files = sorted(output_dir.glob("shard-*.tar"))

    if not shard_files:
        return existing

    print(f"🔍 Scanning {len(shard_files)} existing shards...")

    for shard_path in tqdm(shard_files, desc="Scanning shards", unit="shard"):
        try:
            dataset = wds.WebDataset(str(shard_path))
            for sample in dataset:
                # Read study_id from json metadata (same as DataLoader)
                metadata = json.loads(sample['json'].decode('utf-8'))
                study_id = metadata.get('study_id', '')
                if study_id:
                    if study_id in existing:
                        print(f"   ⚠️  Warning: Duplicate study_id '{study_id}' found in {shard_path.name}")
                    existing.add(study_id)
        except Exception as e:
            print(f"   ⚠️  Warning: Failed to read {shard_path.name}: {e}")
            continue

    print(f"   Found {len(existing)} already processed samples")

    return existing


def infer_samples_per_shard(output_dir: Path) -> int:
    """
    Infer SAMPLES_PER_SHARD from the first shard.

    Args:
        output_dir: Directory containing shard-*.tar files

    Returns:
        Number of samples per shard (default 128 if no shards exist)
    """
    first_shard = output_dir / "shard-000000.tar"

    if not first_shard.exists():
        print(f"⚠️  First shard not found, using default SAMPLES_PER_SHARD=128")
        return 128

    count = 0
    try:
        dataset = wds.WebDataset(str(first_shard))
        for _ in dataset:
            count += 1
    except Exception as e:
        print(f"⚠️  Failed to read first shard: {e}, using default SAMPLES_PER_SHARD=128")
        return 128

    print(f"📊 Inferred SAMPLES_PER_SHARD = {count} (from {first_shard.name})")
    return count


def get_last_shard_info(output_dir: Path) -> Tuple[int, List[Dict], int]:
    """
    Get information about the last shard.

    Args:
        output_dir: Directory containing shard-*.tar files

    Returns:
        Tuple of (last_shard_index, samples_in_last_shard, count_in_last_shard)
        - If no shards exist, returns (-1, [], 0)
        - samples_in_last_shard: List of sample dicts from the last shard
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_files = sorted(output_dir.glob("shard-*.tar"))

    if not shard_files:
        return (-1, [], 0)

    last_shard = shard_files[-1]

    # Extract shard index from filename (e.g., shard-000315.tar -> 315)
    shard_idx = int(last_shard.stem.split('-')[1])

    # Read all samples from last shard
    samples = []
    try:
        dataset = wds.WebDataset(str(last_shard))
        for sample in dataset:
            samples.append(sample)
    except Exception as e:
        print(f"⚠️  Warning: Failed to read last shard {last_shard.name}: {e}")
        return (shard_idx, [], 0)

    count = len(samples)
    print(f"📦 Last shard: {last_shard.name} ({count} samples)")

    return (shard_idx, samples, count)


def generate_manifest(output_dir: Path, split: str) -> Path:
    """
    Generate manifest.json by scanning all existing shards.

    Args:
        output_dir: Output directory
        split: 'train' or 'valid'

    Returns:
        Path to manifest.json
    """
    manifest_path = output_dir / "manifest.json"

    # Count total samples
    total_samples = 0
    shard_info = []

    shard_files = sorted(output_dir.glob("shard-*.tar"))

    for shard_path in shard_files:
        # Extract shard index
        shard_idx = int(shard_path.stem.split('-')[1])

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
            'shard_index': shard_idx,
            'filename': shard_path.name,
            'num_samples': num_samples,
            'size_bytes': shard_path.stat().st_size
        })

    # Generate manifest
    manifest = {
        'dataset': 'CT-RATE',
        'split': split,
        'format': 'webdataset',
        'preprocessed': True,
        'total_shards': len(shard_files),
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
    print(f"   Total shards: {len(shard_files)}")

    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Build preprocessed WebDataset from Hugging Face (Simplified)"
    )
    parser.add_argument('--split', type=str, required=True, choices=['train', 'valid'],
                        help='Dataset split to process')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for preprocessed WebDataset')
    parser.add_argument('--repo-id', type=str, default='ibrahimhamamci/CT-RATE',
                        help='HuggingFace repo ID (default: ibrahimhamamci/CT-RATE)')
    parser.add_argument('--temp-dir', type=str, default=None,
                        help='Temporary directory for downloads (default: system /tmp)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup temp directory
    if args.temp_dir:
        temp_base_dir = Path(args.temp_dir)
        temp_base_dir.mkdir(parents=True, exist_ok=True)
        print(f"🗂️  Using custom temp dir: {temp_base_dir}")
    else:
        temp_base_dir = None
        print(f"🗂️  Using system temp dir: /tmp")

    print("="*80)
    print("Build Preprocessed WebDataset from Hugging Face")
    print("="*80)
    print(f"Repo ID: {args.repo_id}")
    print(f"Split: {args.split}")
    print(f"Output dir: {output_dir}")
    print()

    # 1. Get file list from HuggingFace
    hf_files = get_hf_file_list(args.repo_id, args.split)

    if len(hf_files) == 0:
        print(f"❌ No files found for split '{args.split}'")
        return 1

    # Convert to study_ids
    hf_study_ids = set()
    for file_path in hf_files:
        filename = os.path.basename(file_path)
        study_id = filename.replace('.nii.gz', '')
        hf_study_ids.add(study_id)

    print(f"   Total files on HF: {len(hf_study_ids)}")

    # 2. Scan existing shards
    existing_study_ids = get_existing_samples(output_dir)

    # 3. Calculate missing files
    missing_study_ids = hf_study_ids - existing_study_ids
    missing_files = [f for f in hf_files if os.path.basename(f).replace('.nii.gz', '') in missing_study_ids]

    print(f"\n📊 Status:")
    print(f"   Already processed: {len(existing_study_ids)}")
    print(f"   Missing: {len(missing_study_ids)}")

    if len(missing_study_ids) == 0:
        print("\n✅ All files already processed!")
        generate_manifest(output_dir, args.split)
        return 0

    # 4. Infer SAMPLES_PER_SHARD
    samples_per_shard = infer_samples_per_shard(output_dir)

    # 5. Get last shard info
    last_shard_idx, last_shard_samples, last_shard_count = get_last_shard_info(output_dir)

    # 6. Load CSV files
    print(f"\n📚 Loading CSV files...")
    with managed_temp_dir(temp_base_dir, f"ct_rate_csv") as csv_temp_dir:
        csv_temp_path = Path(csv_temp_dir)
        metadata_df, labels_df, reports_df = load_csv_files(
            args.repo_id,
            args.split,
            csv_temp_path
        )

    print(f"   Loaded metadata: {len(metadata_df)} records")
    print(f"   Loaded labels: {len(labels_df)} records")
    print(f"   Loaded reports: {len(reports_df)} records")

    # 7. Process missing files (single-threaded)
    print(f"\n🔄 Processing {len(missing_files)} missing files...")

    with managed_temp_dir(temp_base_dir, f"ct_rate_{args.split}") as temp_dir:
        temp_path = Path(temp_dir)

        # Prepare for writing
        current_shard_idx = last_shard_idx + 1 if last_shard_idx >= 0 else 0
        current_samples = list(last_shard_samples)  # Copy samples from last shard if incomplete

        # If last shard was incomplete, we'll rewrite it
        if last_shard_count > 0 and last_shard_count < samples_per_shard:
            current_shard_idx = last_shard_idx
            print(f"   ♻️  Will append to incomplete shard-{current_shard_idx:06d} (currently {last_shard_count}/{samples_per_shard})")

        processed_count = 0
        failed_count = 0

        pbar = tqdm(missing_files, desc="Processing files", unit="file")

        for hf_file_path in pbar:
            filename = os.path.basename(hf_file_path)
            study_id = filename.replace('.nii.gz', '')

            try:
                # Download file
                local_path = hf_hub_download(
                    repo_id=args.repo_id,
                    filename=hf_file_path,
                    repo_type="dataset",
                    local_dir=temp_path,
                    local_dir_use_symlinks=False
                )

                # Load nii.gz file with nibabel
                nii_img = nib.load(str(local_path))
                volume_data = nii_img.get_fdata().astype(np.float32)  # (H, W, D)

                # Get spacing from NIfTI header
                header = nii_img.header
                pixdim = header['pixdim'][1:4]  # [x_spacing, y_spacing, z_spacing]
                xy_spacing = float(pixdim[0])
                z_spacing = float(pixdim[2])

                # Get metadata from CSV
                if study_id not in metadata_df.index:
                    print(f"\n   ⚠️  Skipping {study_id}: not found in metadata")
                    failed_count += 1
                    continue

                meta_row = metadata_df.loc[study_id]
                slope = float(meta_row.get('RescaleSlope', 1.0))
                intercept = float(meta_row.get('RescaleIntercept', 0.0))

                # Get labels from CSV
                if study_id not in labels_df.index:
                    print(f"\n   ⚠️  Skipping {study_id}: not found in labels")
                    failed_count += 1
                    continue

                label_row = labels_df.loc[study_id]
                disease_labels = label_row.values.astype(np.float32)

                # Get report from CSV
                if study_id not in reports_df.index:
                    print(f"\n   ⚠️  Skipping {study_id}: not found in reports")
                    failed_count += 1
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

                # Create sample
                sample = {
                    '__key__': study_id,
                    'bin': preprocessed_volume.tobytes(),
                    'txt': report_text,
                    'labels': disease_labels.tobytes(),  # Changed from 'cls' to 'labels'
                    'json': json.dumps({
                        'study_id': study_id,
                        'shape': list(preprocessed_volume.shape),
                        'dtype': 'float16',
                        'num_classes': len(disease_labels)
                    })
                }

                # Add to current samples
                current_samples.append(sample)
                processed_count += 1

                # Delete downloaded file
                os.remove(local_path)

                # Write shard if full
                if len(current_samples) >= samples_per_shard:
                    shard_path = output_dir / f"shard-{current_shard_idx:06d}.tar"
                    with wds.TarWriter(str(shard_path)) as sink:
                        for s in current_samples:
                            sink.write(s)

                    pbar.set_postfix({
                        'shard': current_shard_idx,
                        'processed': processed_count,
                        'failed': failed_count
                    })

                    # Reset for next shard
                    current_shard_idx += 1
                    current_samples = []

            except Exception as e:
                print(f"\n❌ Failed to process {study_id}: {e}")
                failed_count += 1
                continue

        # Write remaining samples
        if len(current_samples) > 0:
            shard_path = output_dir / f"shard-{current_shard_idx:06d}.tar"
            with wds.TarWriter(str(shard_path)) as sink:
                for s in current_samples:
                    sink.write(s)
            print(f"\n   ✅ Wrote final shard: {shard_path.name} ({len(current_samples)} samples)")

        pbar.close()

    # 8. Generate manifest
    generate_manifest(output_dir, args.split)

    # 9. Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"✅ Processed: {processed_count} files")
    print(f"❌ Failed: {failed_count} files")
    print(f"📁 Output: {output_dir}")
    print("="*80)

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    exit(main())
