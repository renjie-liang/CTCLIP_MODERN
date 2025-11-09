"""
Convert NPZ files to WebDataset format with float16 compression.

This script converts CT volume data from NPZ (float32) to WebDataset TAR format (float16),
significantly reducing storage requirements while maintaining acceptable precision.

Usage:
    python scripts/convert_npz_to_webdataset.py \
        --data_folder /path/to/npz/files \
        --reports_file /path/to/reports.csv \
        --meta_file /path/to/metadata.csv \
        --labels_file /path/to/labels.csv \
        --output_dir /path/to/output \
        --samples_per_shard 100 \
        --num_workers 8
"""

import os
import sys
import argparse
import json
import io
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import tarfile
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob


def load_metadata(reports_file: str, meta_file: str, labels_file: str) -> Tuple[Dict, pd.DataFrame, Dict]:
    """Load all metadata files and index by study_id."""

    # Load reports
    df_reports = pd.read_csv(reports_file)
    reports_dict = {}
    for _, row in df_reports.iterrows():
        study_id = str(row['VolumeName']).replace(".nii.gz", "")
        findings = row.get("Findings_EN", "")
        impressions = row.get("Impressions_EN", "")
        reports_dict[study_id] = {
            'findings': str(findings),
            'impressions': str(impressions)
        }

    # Load metadata
    df_meta = pd.read_csv(meta_file)
    df_meta['study_id'] = df_meta['VolumeName'].str.replace(".nii.gz", "", regex=False)
    df_meta = df_meta.set_index('study_id')

    # Load labels
    df_labels = pd.read_csv(labels_file)
    df_labels['study_id'] = df_labels['VolumeName'].str.replace(".nii.gz", "", regex=False)
    label_cols = [col for col in df_labels.columns if col not in ['VolumeName', 'study_id']]

    labels_dict = {}
    for _, row in df_labels.iterrows():
        study_id = row['study_id']
        labels = row[label_cols].values.astype(np.float32)
        labels_dict[study_id] = labels

    print(f"Loaded metadata:")
    print(f"  - Reports: {len(reports_dict)}")
    print(f"  - Metadata: {len(df_meta)}")
    print(f"  - Labels: {len(labels_dict)} ({len(label_cols)} classes)")

    return reports_dict, df_meta, labels_dict


def find_npz_files(data_folder: str) -> List[str]:
    """Find all .npz files in the data folder (supports multiple directory structures)."""

    print(f"Scanning for .npz files in: {data_folder}")

    # Try 3-layer structure (patient/accession/*.npz)
    npz_files_3layer = glob.glob(os.path.join(data_folder, '*', '*', '*.npz'))

    # Try 2-layer structure (patient/*.npz)
    npz_files_2layer = glob.glob(os.path.join(data_folder, '*', '*.npz'))

    # Try 1-layer structure (*.npz)
    npz_files_1layer = glob.glob(os.path.join(data_folder, '*.npz'))

    # Pick the structure with most files
    structure_options = [
        (len(npz_files_3layer), "3-layer", npz_files_3layer),
        (len(npz_files_2layer), "2-layer", npz_files_2layer),
        (len(npz_files_1layer), "1-layer", npz_files_1layer)
    ]
    structure_options.sort(reverse=True, key=lambda x: x[0])

    num_files, structure_name, npz_files = structure_options[0]
    print(f"Detected {structure_name} structure: {num_files} files")

    return npz_files


def process_single_sample(
    npz_path: str,
    study_id: str,
    reports_dict: Dict,
    meta_df: pd.DataFrame,
    labels_dict: Dict
) -> Dict[str, bytes]:
    """
    Process a single NPZ file and return WebDataset-compatible data.

    Returns:
        Dict with keys: 'npy', 'json', 'txt', 'labels'
    """

    # Load NPZ data (use mmap for memory efficiency)
    npz_data = np.load(npz_path, mmap_mode='r')["data"]

    # Convert to float16 (this is the key space savings!)
    # Note: We store raw data here, processing will be done during training
    # Ensure the array is C-contiguous for proper serialization
    volume_fp16 = np.ascontiguousarray(npz_data.astype(np.float16))

    # Serialize volume to bytes
    volume_bytes = io.BytesIO()
    np.save(volume_bytes, volume_fp16)
    volume_bytes = volume_bytes.getvalue()

    # Prepare metadata JSON
    meta_row = meta_df.loc[study_id]
    metadata = {
        'study_id': study_id,
        'RescaleSlope': float(meta_row["RescaleSlope"]),
        'RescaleIntercept': float(meta_row["RescaleIntercept"]),
        'XYSpacing': str(meta_row["XYSpacing"]),
        'ZSpacing': float(meta_row["ZSpacing"]),
        'original_shape': list(npz_data.shape)
    }
    metadata_bytes = json.dumps(metadata).encode('utf-8')

    # Prepare report text
    report = reports_dict[study_id]
    report_text = f"{report['findings']}\n{report['impressions']}"
    report_bytes = report_text.encode('utf-8')

    # Prepare labels
    labels = labels_dict[study_id]
    labels_bytes = labels.tobytes()

    return {
        'npy': volume_bytes,      # Volume data (float16)
        'json': metadata_bytes,   # Metadata
        'txt': report_bytes,      # Report text
        'labels': labels_bytes    # Disease labels (use .labels to avoid auto-decoding)
    }


def create_shard(
    shard_id: int,
    samples: List[Tuple[str, str]],
    output_dir: Path,
    reports_dict: Dict,
    meta_df: pd.DataFrame,
    labels_dict: Dict
) -> Tuple[int, int, int]:
    """
    Create a single WebDataset shard (TAR file).

    Returns:
        (shard_id, num_samples, shard_size_bytes)
    """

    shard_path = output_dir / f"shard-{shard_id:06d}.tar"

    num_samples = 0
    total_size = 0

    with tarfile.open(shard_path, 'w') as tar:
        for idx, (npz_path, study_id) in enumerate(samples):
            try:
                # Process sample
                sample_data = process_single_sample(
                    npz_path, study_id, reports_dict, meta_df, labels_dict
                )

                # Add each component to TAR with WebDataset naming convention
                # Format: {shard_id:06d}_{idx:06d}.{ext}
                base_name = f"{shard_id:06d}_{idx:06d}"

                for ext, data in sample_data.items():
                    tarinfo = tarfile.TarInfo(name=f"{base_name}.{ext}")
                    tarinfo.size = len(data)
                    tar.addfile(tarinfo, io.BytesIO(data))
                    total_size += len(data)

                num_samples += 1

            except Exception as e:
                print(f"Error processing {study_id}: {e}")
                continue

    shard_size_mb = total_size / (1024**2)
    return shard_id, num_samples, shard_size_mb


def main():
    parser = argparse.ArgumentParser(description="Convert NPZ to WebDataset format")
    parser.add_argument('--data_folder', type=str, required=True, help='Folder containing NPZ files')
    parser.add_argument('--reports_file', type=str, required=True, help='CSV file with reports')
    parser.add_argument('--meta_file', type=str, required=True, help='CSV file with metadata')
    parser.add_argument('--labels_file', type=str, required=True, help='CSV file with labels')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for shards')
    parser.add_argument('--samples_per_shard', type=int, default=100, help='Number of samples per shard')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--test_mode', action='store_true', help='Test mode: only convert first 100 samples')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("NPZ to WebDataset Conversion (float32 → float16)")
    print("="*80)

    # Load metadata
    print("\n[1/4] Loading metadata...")
    reports_dict, meta_df, labels_dict = load_metadata(
        args.reports_file, args.meta_file, args.labels_file
    )

    # Find NPZ files
    print("\n[2/4] Finding NPZ files...")
    npz_files = find_npz_files(args.data_folder)

    if len(npz_files) == 0:
        print("ERROR: No NPZ files found!")
        return

    # Match with metadata
    print("\n[3/4] Matching files with metadata...")
    matched_samples = []
    for npz_path in tqdm(npz_files, desc="Matching"):
        study_id = Path(npz_path).stem

        if (study_id in reports_dict and
            study_id in meta_df.index and
            study_id in labels_dict):
            matched_samples.append((npz_path, study_id))

    print(f"Matched {len(matched_samples)} / {len(npz_files)} samples")

    if args.test_mode:
        print("\n⚠️  TEST MODE: Only converting first 100 samples")
        matched_samples = matched_samples[:100]

    # Create shards
    print(f"\n[4/4] Creating WebDataset shards...")
    print(f"  - Samples per shard: {args.samples_per_shard}")
    print(f"  - Parallel workers: {args.num_workers}")

    # Split into shards
    shards = []
    for i in range(0, len(matched_samples), args.samples_per_shard):
        shard_samples = matched_samples[i:i+args.samples_per_shard]
        shard_id = i // args.samples_per_shard
        shards.append((shard_id, shard_samples))

    print(f"  - Total shards: {len(shards)}")

    # Process shards in parallel
    total_samples = 0
    total_size_mb = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                create_shard,
                shard_id,
                shard_samples,
                output_dir,
                reports_dict,
                meta_df,
                labels_dict
            ): shard_id
            for shard_id, shard_samples in shards
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Creating shards"):
            shard_id, num_samples, shard_size_mb = future.result()
            total_samples += num_samples
            total_size_mb += shard_size_mb

    # Write manifest file
    manifest = {
        'num_shards': len(shards),
        'samples_per_shard': args.samples_per_shard,
        'total_samples': total_samples,
        'total_size_mb': total_size_mb,
        'total_size_gb': total_size_mb / 1024,
        'average_sample_size_mb': total_size_mb / total_samples if total_samples > 0 else 0
    }

    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Summary
    print("\n" + "="*80)
    print("CONVERSION COMPLETE!")
    print("="*80)
    print(f"Total samples converted: {total_samples}")
    print(f"Total shards created: {len(shards)}")
    print(f"Total size: {total_size_mb/1024:.2f} GB")
    print(f"Average sample size: {total_size_mb/total_samples:.2f} MB")
    print(f"Output directory: {output_dir}")
    print(f"Manifest file: {manifest_path}")

    # Estimate compression ratio
    original_size_estimate = len(matched_samples) * 350  # MB (based on your 14TB / 40k samples)
    compression_ratio = original_size_estimate / total_size_mb if total_size_mb > 0 else 0
    print(f"\nEstimated compression ratio: {compression_ratio:.2f}x")
    print(f"Storage savings: {(1 - 1/compression_ratio)*100:.1f}%")


if __name__ == '__main__':
    main()
