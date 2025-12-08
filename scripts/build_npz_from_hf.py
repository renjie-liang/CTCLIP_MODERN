#!/usr/bin/env python3
"""
Build NPZ dataset from HuggingFace nii.gz files.

This script:
1. Downloads nii.gz files from HuggingFace (if not already downloaded)
2. Preprocesses: orientation, spacing, crop/pad
3. Saves as NPZ with complete metadata (affine, crop_bbox, etc.)
4. Generates manifest.json

Usage:
    python scripts/build_npz_from_hf.py --split valid --max-files 20
    python scripts/build_npz_from_hf.py --split train
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
import random

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from tqdm import tqdm
from huggingface_hub import list_repo_files, hf_hub_download

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import configuration
from config_npz_conversion import (
    HF_REPO_ID,
    HF_REPO_TYPE,
    TARGET_SPACING,
    TARGET_SHAPE,
    HU_CLIP_MIN,
    HU_CLIP_MAX,
    PAD_VALUE,
    STORAGE_DTYPE,
    TARGET_ORIENTATION,
    USE_NESTED_STRUCTURE,
    LOCAL_SOURCE_DIRS,
    TEMP_DIR,
    DELETE_SOURCE_AFTER_CONVERSION,
    SKIP_EXISTING,
    CHECK_ORIENTATION,
    VALIDATE_SPACING,
    SPACING_TOLERANCE,
    VERBOSE,
    PROGRESS_INTERVAL,
    SPLIT_CONFIGS
)


# ============================================================================
# Utility Functions
# ============================================================================

def scan_local_nii_files(local_dir: Path) -> Dict[str, Path]:
    """
    Recursively scan local directory for nii.gz files.

    Args:
        local_dir: Local directory to scan

    Returns:
        Dictionary mapping study_id to local file path
    """
    if not local_dir.exists():
        return {}

    print(f"\n🔍 Scanning local directory: {local_dir}")

    # Recursively find all nii.gz files
    nii_files = list(local_dir.rglob("*.nii.gz"))

    # Build study_id -> path mapping
    local_files = {}
    for nii_path in nii_files:
        study_id = nii_path.stem.replace('.nii', '')
        local_files[study_id] = nii_path

    print(f"   Found {len(local_files)} local nii.gz files")

    return local_files


def extract_study_id(hf_path: str) -> str:
    """
    Extract study_id from HF path.

    Example:
        'dataset/train_fixed/train_001/train_001_a/train_001_a_1.nii.gz'
        -> 'train_001_a_1'
    """
    filename = Path(hf_path).stem  # Remove .nii.gz
    return filename.replace('.nii', '')


def get_patient_id(study_id: str) -> str:
    """
    Extract patient ID from study_id for nested directory structure.

    Example:
        'train_001_a_1' -> 'train_001'
    """
    parts = study_id.split('_')
    # Assuming format: {split}_{patient}_{series}_{scan}
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"  # e.g., 'train_001'
    return study_id


def check_and_fix_orientation(
    nii_img: nib.Nifti1Image,
    target_orient: str = "LPS"
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Check orientation and fix if needed.

    Args:
        nii_img: NIfTI image
        target_orient: Target orientation (e.g., 'LPS')

    Returns:
        (data, affine, current_orientation_code)
    """
    # Get current orientation
    current_orient = nib.orientations.io_orientation(nii_img.affine)
    current_orient_code = ''.join(nib.orientations.ornt2axcodes(current_orient))

    if VERBOSE:
        print(f"      Current orientation: {current_orient_code}")

    # Check if already in target orientation
    if current_orient_code == target_orient:
        if VERBOSE:
            print(f"      ✓ Already in {target_orient} orientation")
        data = nii_img.get_fdata().astype(np.float32)
        return data, nii_img.affine.copy(), current_orient_code

    # Need to reorient
    if VERBOSE:
        print(f"      → Reorienting from {current_orient_code} to {target_orient}")

    # Calculate transformation
    target_orient_code = nib.orientations.axcodes2ornt(target_orient)
    transform = nib.orientations.ornt_transform(current_orient, target_orient_code)

    # Apply transformation to data
    data = nii_img.get_fdata().astype(np.float32)
    data_reoriented = nib.orientations.apply_orientation(data, transform)

    # Update affine matrix
    # Create affine transformation matrix for the reorientation
    affine_reoriented = nii_img.affine @ nib.orientations.inv_ornt_aff(transform, data.shape)

    return data_reoriented, affine_reoriented, target_orient


def resample_volume(
    data: np.ndarray,
    current_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
    affine: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample volume to target spacing using trilinear interpolation.

    Args:
        data: Volume data (H, W, D)
        current_spacing: Current voxel spacing [z, y, x] in mm
        target_spacing: Target voxel spacing [z, y, x] in mm
        affine: Current affine matrix (4, 4)

    Returns:
        (resampled_data, updated_affine)
    """
    if VERBOSE:
        print(f"      Current spacing: {current_spacing}")
        print(f"      Target spacing: {target_spacing}")

    # Transpose to (D, H, W) for PyTorch interpolation
    data_dhw = data.transpose(2, 0, 1)  # (H, W, D) -> (D, H, W)

    # Calculate scaling factors
    scaling_factors = [
        current_spacing[i] / target_spacing[i]
        for i in range(3)
    ]

    # Calculate new shape
    original_shape = data_dhw.shape
    new_shape = [
        int(original_shape[i] * scaling_factors[i])
        for i in range(3)
    ]

    if VERBOSE:
        print(f"      Original shape (D,H,W): {original_shape}")
        print(f"      New shape (D,H,W): {new_shape}")

    # Convert to tensor and add batch/channel dimensions
    tensor = torch.from_numpy(data_dhw).float()
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

    # Resample
    resampled = F.interpolate(
        tensor,
        size=new_shape,
        mode='trilinear',
        align_corners=False
    )

    # Remove batch/channel dimensions and transpose back
    resampled = resampled.squeeze(0).squeeze(0)  # (D, H, W)
    resampled_hw = resampled.permute(1, 2, 0)  # (D, H, W) -> (H, W, D)
    resampled_np = resampled_hw.numpy()

    # Update affine matrix for new spacing
    # The scaling changes the voxel size, so we update the affine
    new_affine = affine.copy()
    for i in range(3):
        new_affine[:3, i] = affine[:3, i] * (current_spacing[i] / target_spacing[i])

    # Validate spacing
    if VALIDATE_SPACING:
        new_spacing = np.abs(new_affine[:3, :3]).max(axis=0)
        spacing_diff = np.abs(np.array(target_spacing) - new_spacing)
        if np.any(spacing_diff > SPACING_TOLERANCE):
            print(f"      ⚠️ Warning: Spacing validation failed")
            print(f"         Expected: {target_spacing}")
            print(f"         Got: {new_spacing}")

    return resampled_np, new_affine


def crop_or_pad_volume(
    data: np.ndarray,
    target_shape: Tuple[int, int, int],
    pad_value: float,
    affine: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
    """
    Crop or pad volume to target shape.

    Args:
        data: Volume data (H, W, D)
        target_shape: Target shape (D, H, W)
        pad_value: Value to use for padding
        affine: Current affine matrix (4, 4)

    Returns:
        (processed_data, updated_affine, crop_bbox, pad_params)
    """
    current_shape = data.shape  # (H, W, D)
    target_D, target_H, target_W = target_shape

    H, W, D = current_shape

    if VERBOSE:
        print(f"      Current shape (H,W,D): {current_shape}")
        print(f"      Target shape (D,H,W): {target_shape}")

    # Initialize tracking
    crop_bbox = {'z': [0, D], 'y': [0, H], 'x': [0, W]}
    pad_params = {'z': [0, 0], 'y': [0, 0], 'x': [0, 0]}

    # Transpose to (D, H, W) for easier processing
    data_dhw = data.transpose(2, 0, 1)

    # Process each dimension
    # Depth (D/Z dimension)
    if D > target_D:
        # Crop
        start_z = (D - target_D) // 2
        end_z = start_z + target_D
        data_dhw = data_dhw[start_z:end_z, :, :]
        crop_bbox['z'] = [start_z, end_z]
    elif D < target_D:
        # Pad
        pad_total = target_D - D
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        data_dhw = np.pad(
            data_dhw,
            ((pad_before, pad_after), (0, 0), (0, 0)),
            mode='constant',
            constant_values=pad_value
        )
        pad_params['z'] = [pad_before, pad_after]

    # Height (H/Y dimension)
    _, H_cur, _ = data_dhw.shape
    if H_cur > target_H:
        start_y = (H_cur - target_H) // 2
        end_y = start_y + target_H
        data_dhw = data_dhw[:, start_y:end_y, :]
        crop_bbox['y'] = [start_y, end_y]
    elif H_cur < target_H:
        pad_total = target_H - H_cur
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        data_dhw = np.pad(
            data_dhw,
            ((0, 0), (pad_before, pad_after), (0, 0)),
            mode='constant',
            constant_values=pad_value
        )
        pad_params['y'] = [pad_before, pad_after]

    # Width (W/X dimension)
    _, _, W_cur = data_dhw.shape
    if W_cur > target_W:
        start_x = (W_cur - target_W) // 2
        end_x = start_x + target_W
        data_dhw = data_dhw[:, :, start_x:end_x]
        crop_bbox['x'] = [start_x, end_x]
    elif W_cur < target_W:
        pad_total = target_W - W_cur
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        data_dhw = np.pad(
            data_dhw,
            ((0, 0), (0, 0), (pad_before, pad_after)),
            mode='constant',
            constant_values=pad_value
        )
        pad_params['x'] = [pad_before, pad_after]

    # Transpose back to (H, W, D)
    data_final = data_dhw.transpose(1, 2, 0)

    # Update affine for cropping (cropping changes origin)
    new_affine = affine.copy()

    # Calculate offset in voxel coordinates
    offset_voxels = np.array([
        crop_bbox['x'][0],  # X offset
        crop_bbox['y'][0],  # Y offset
        crop_bbox['z'][0],  # Z offset
        1
    ])

    # Transform to world coordinates and update origin
    offset_world = affine @ offset_voxels
    new_affine[:3, 3] = offset_world[:3]

    if VERBOSE:
        print(f"      Crop bbox: {crop_bbox}")
        print(f"      Pad params: {pad_params}")
        print(f"      Final shape (H,W,D): {data_final.shape}")

    return data_final, new_affine, crop_bbox, pad_params


def generate_metadata(
    study_id: str,
    source_file: str,
    original_affine: np.ndarray,
    final_affine: np.ndarray,
    orientation: str,
    final_spacing: List[float],
    original_shape: Tuple,
    final_shape: Tuple,
    crop_bbox: Dict,
    pad_params: Dict,
    hu_stats: Dict
) -> Dict:
    """Generate complete metadata dictionary."""
    metadata = {
        # Identity
        'study_id': study_id,
        'source_file': source_file,

        # Spatial geometry (critical for reconstruction)
        'affine': {
            'original': original_affine.tolist(),
            'final': final_affine.tolist()
        },
        'orientation': orientation,
        'spacing': final_spacing,

        # Shape information
        'shape': {
            'original': list(original_shape),
            'final': list(final_shape)
        },

        # Crop/Pad information (for reversing)
        'crop_bbox': crop_bbox,
        'pad_params': pad_params,

        # Preprocessing parameters
        'preprocessing': {
            'clip_range': [HU_CLIP_MIN, HU_CLIP_MAX],
            'target_spacing': TARGET_SPACING,
            'target_shape': list(TARGET_SHAPE),
            'pad_value': PAD_VALUE,
            'dtype': STORAGE_DTYPE
        },

        # Quality metrics
        'quality': hu_stats
    }

    return metadata


def save_npz(
    volume: np.ndarray,
    metadata: Dict,
    output_path: Path
):
    """
    Save volume and metadata to NPZ file.

    Args:
        volume: Volume data (H, W, D) in int16
        metadata: Metadata dictionary
        output_path: Output NPZ file path
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert metadata to JSON string
    metadata_json = json.dumps(metadata, indent=2)

    # Save NPZ (compressed)
    np.savez_compressed(
        output_path,
        volume=volume,
        metadata=metadata_json.encode('utf-8')
    )

    # Get file size
    size_mb = output_path.stat().st_size / (1024**2)

    if VERBOSE:
        print(f"      ✓ Saved to: {output_path}")
        print(f"      Size: {size_mb:.2f} MB")

    return size_mb


# ============================================================================
# Main Processing Functions
# ============================================================================

def preprocess_single_file(
    hf_file_path: str,
    output_dir: Path,
    temp_dir: Path,
    local_nii_path: Path = None
) -> Dict:
    """
    Process a single nii.gz file.

    Args:
        hf_file_path: HuggingFace file path (for reference)
        output_dir: Output directory for NPZ
        temp_dir: Temp directory for downloads
        local_nii_path: Local nii.gz path if already downloaded

    Returns:
        Result dictionary with statistics
    """
    study_id = extract_study_id(hf_file_path)

    if VERBOSE:
        print(f"\n{'='*80}")
        print(f"Processing: {study_id}")
        print(f"{'='*80}")

    # Determine output path
    if USE_NESTED_STRUCTURE:
        patient_id = get_patient_id(study_id)
        npz_path = output_dir / patient_id / f"{study_id}.npz"
    else:
        npz_path = output_dir / f"{study_id}.npz"

    # Skip if already exists
    if SKIP_EXISTING and npz_path.exists():
        if VERBOSE:
            print(f"   ⏭️  Skipping: NPZ already exists")
        return {
            'study_id': study_id,
            'status': 'skipped',
            'npz_path': str(npz_path)
        }

    # Get nii.gz file path
    source_was_local = False
    need_cleanup = False

    if local_nii_path and local_nii_path.exists():
        # Use local file
        if VERBOSE:
            print(f"   [1] Using local file...")
            print(f"      ✓ Local path: {local_nii_path}")
        nii_path_to_process = str(local_nii_path)
        source_was_local = True
        # Will delete if configured
        need_cleanup = DELETE_SOURCE_AFTER_CONVERSION
    else:
        # Download from HuggingFace
        if VERBOSE:
            print(f"   [1] Downloading from HuggingFace...")

        nii_path_to_process = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=hf_file_path,
            repo_type=HF_REPO_TYPE,
            local_dir=temp_dir,
            local_dir_use_symlinks=False
        )

        if VERBOSE:
            print(f"      ✓ Downloaded to: {nii_path_to_process}")
        # Always delete downloaded files after processing
        need_cleanup = True

    # Load NIfTI
    if VERBOSE:
        print(f"   [2] Loading NIfTI...")

    nii_img = nib.load(nii_path_to_process)
    original_affine = nii_img.affine.copy()
    original_spacing = nii_img.header.get_zooms()[:3]  # (x, y, z)
    original_spacing_zyx = [original_spacing[2], original_spacing[1], original_spacing[0]]

    if VERBOSE:
        print(f"      Original shape: {nii_img.shape}")
        print(f"      Original spacing (x,y,z): {original_spacing}")

    # Check and fix orientation
    if VERBOSE:
        print(f"   [3] Checking orientation...")

    if CHECK_ORIENTATION:
        data, affine_after_orient, orientation = check_and_fix_orientation(
            nii_img,
            target_orient=TARGET_ORIENTATION
        )
    else:
        data = nii_img.get_fdata().astype(np.float32)
        affine_after_orient = original_affine.copy()
        orientation = TARGET_ORIENTATION

    original_shape = data.shape

    # Clip HU values
    if VERBOSE:
        print(f"   [4] Clipping HU values...")
        print(f"      Range before clip: [{np.min(data):.1f}, {np.max(data):.1f}]")

    data = np.clip(data, HU_CLIP_MIN, HU_CLIP_MAX)

    if VERBOSE:
        print(f"      Range after clip: [{np.min(data):.1f}, {np.max(data):.1f}]")

    # Resample to target spacing
    if VERBOSE:
        print(f"   [5] Resampling to target spacing...")

    data_resampled, affine_after_resample = resample_volume(
        data,
        current_spacing=original_spacing_zyx,
        target_spacing=TARGET_SPACING,
        affine=affine_after_orient
    )

    # Crop or pad to target shape
    if VERBOSE:
        print(f"   [6] Cropping/Padding to target shape...")

    data_final, final_affine, crop_bbox, pad_params = crop_or_pad_volume(
        data_resampled,
        target_shape=TARGET_SHAPE,
        pad_value=PAD_VALUE,
        affine=affine_after_resample
    )

    # Convert to int16
    if VERBOSE:
        print(f"   [7] Converting to {STORAGE_DTYPE}...")

    volume_int16 = data_final.astype(np.int16)

    # Calculate quality metrics
    hu_stats = {
        'hu_min': int(np.min(volume_int16)),
        'hu_max': int(np.max(volume_int16)),
        'hu_mean': float(np.mean(volume_int16)),
        'hu_std': float(np.std(volume_int16)),
        'has_invalid': bool(np.any(np.isnan(volume_int16)))
    }

    if VERBOSE:
        print(f"      HU stats: min={hu_stats['hu_min']}, max={hu_stats['hu_max']}")

    # Generate metadata
    if VERBOSE:
        print(f"   [8] Generating metadata...")

    metadata = generate_metadata(
        study_id=study_id,
        source_file=hf_file_path,
        original_affine=original_affine,
        final_affine=final_affine,
        orientation=orientation,
        final_spacing=TARGET_SPACING,
        original_shape=original_shape,
        final_shape=data_final.shape,
        crop_bbox=crop_bbox,
        pad_params=pad_params,
        hu_stats=hu_stats
    )

    # Save NPZ
    if VERBOSE:
        print(f"   [9] Saving NPZ...")

    size_mb = save_npz(volume_int16, metadata, npz_path)

    # Cleanup source file if needed
    if need_cleanup:
        if VERBOSE:
            print(f"   [10] Cleaning up source file...")
        Path(nii_path_to_process).unlink()
        if VERBOSE:
            source_type = "local" if source_was_local else "downloaded"
            print(f"      ✓ Deleted {source_type} file: {nii_path_to_process}")

    return {
        'study_id': study_id,
        'status': 'success',
        'npz_path': str(npz_path),
        'source_file': hf_file_path,
        'size_mb': size_mb,
        'hu_stats': hu_stats,
        'source_was_local': source_was_local
    }


def list_hf_files(split: str) -> List[str]:
    """List all nii.gz files from HuggingFace for a given split."""
    print(f"\n📋 Listing files from {HF_REPO_ID} (split={split})...")

    all_files = list_repo_files(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE)

    # Filter by split
    path_pattern = SPLIT_CONFIGS[split]['hf_path_pattern']
    split_files = [
        f for f in all_files
        if f.startswith(path_pattern) and f.endswith('.nii.gz')
    ]

    print(f"   Found {len(split_files)} {split} files on HuggingFace")

    return sorted(split_files)


def scan_existing_npz(output_dir: Path) -> Set[str]:
    """Scan existing NPZ files to support resume."""
    if not output_dir.exists():
        return set()

    print(f"\n🔍 Scanning existing NPZ files in {output_dir}...")

    npz_files = list(output_dir.rglob("*.npz"))
    existing = set()

    for npz_path in npz_files:
        # Extract study_id from filename
        study_id = npz_path.stem
        existing.add(study_id)

    print(f"   Found {len(existing)} existing NPZ files")

    return existing


def generate_manifest(output_dir: Path, split: str, results: List[Dict]):
    """Generate manifest.json file."""
    manifest_path = output_dir / "manifest.json"

    # Calculate statistics
    successful = [r for r in results if r['status'] == 'success']
    skipped = [r for r in results if r['status'] == 'skipped']
    failed = [r for r in results if r['status'] == 'failed']

    total_size_mb = sum(r.get('size_mb', 0) for r in successful)

    manifest = {
        'dataset': 'CT-RATE',
        'split': split,
        'format': 'npz',
        'version': '1.0',

        'preprocessing': {
            'clip_range': [HU_CLIP_MIN, HU_CLIP_MAX],
            'target_spacing': TARGET_SPACING,
            'target_shape': list(TARGET_SHAPE),
            'orientation': TARGET_ORIENTATION,
            'pad_value': PAD_VALUE,
            'dtype': STORAGE_DTYPE
        },

        'statistics': {
            'total_files': len(results),
            'successful': len(successful),
            'skipped': len(skipped),
            'failed': len(failed),
            'total_size_mb': round(total_size_mb, 2),
            'total_size_gb': round(total_size_mb / 1024, 2)
        },

        'files': results
    }

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n📄 Manifest saved to: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build NPZ dataset from HuggingFace nii.gz files"
    )
    parser.add_argument(
        '--split',
        type=str,
        required=True,
        choices=['train', 'valid'],
        help='Dataset split to process'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Maximum number of files to process (for testing)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for sampling'
    )

    args = parser.parse_args()

    # Get configuration
    output_dir = Path(SPLIT_CONFIGS[args.split]['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(TEMP_DIR) if TEMP_DIR else Path('/tmp/ct_rate_conversion')
    temp_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Build NPZ Dataset from HuggingFace")
    print("="*80)
    print(f"Split: {args.split}")
    print(f"Output dir: {output_dir}")
    print(f"Temp dir: {temp_dir}")
    print(f"Target spacing: {TARGET_SPACING}")
    print(f"Target shape: {TARGET_SHAPE}")
    print(f"HU clip range: [{HU_CLIP_MIN}, {HU_CLIP_MAX}]")
    print(f"Delete source files: {DELETE_SOURCE_AFTER_CONVERSION}")

    # Scan local nii.gz files (already downloaded)
    local_source_dir = Path(LOCAL_SOURCE_DIRS[args.split])
    local_files_map = scan_local_nii_files(local_source_dir)

    # List HF files
    hf_files = list_hf_files(args.split)

    if len(hf_files) == 0:
        print(f"\n❌ No files found for split '{args.split}'")
        return 1

    # Scan existing NPZ files
    existing_study_ids = scan_existing_npz(output_dir)

    # Calculate missing files
    missing_files = []
    for hf_path in hf_files:
        study_id = extract_study_id(hf_path)
        if study_id not in existing_study_ids:
            missing_files.append(hf_path)

    print(f"\n📊 Status:")
    print(f"   Total files on HF: {len(hf_files)}")
    print(f"   Already processed: {len(existing_study_ids)}")
    print(f"   Missing: {len(missing_files)}")

    # Apply max_files limit
    files_to_process = missing_files
    if args.max_files is not None:
        random.seed(args.random_seed)
        if len(files_to_process) > args.max_files:
            files_to_process = random.sample(files_to_process, args.max_files)
            print(f"\n⚠️  Limiting to {args.max_files} files for testing")

    print(f"\n🚀 Processing {len(files_to_process)} files...")

    # Count how many files are available locally
    local_available = 0
    for hf_path in files_to_process:
        study_id = extract_study_id(hf_path)
        if study_id in local_files_map:
            local_available += 1

    print(f"   Files available locally: {local_available}/{len(files_to_process)}")
    print(f"   Files to download: {len(files_to_process) - local_available}/{len(files_to_process)}")

    # Process files
    results = []
    for idx, hf_path in enumerate(files_to_process, 1):
        if not VERBOSE and idx % PROGRESS_INTERVAL == 0:
            print(f"   Progress: {idx}/{len(files_to_process)}")

        # Check if local file exists
        study_id = extract_study_id(hf_path)
        local_path = local_files_map.get(study_id, None)

        result = preprocess_single_file(hf_path, output_dir, temp_dir, local_path)
        results.append(result)

    # Generate manifest
    generate_manifest(output_dir, args.split, results)

    # Summary
    successful = [r for r in results if r['status'] == 'success']
    skipped = [r for r in results if r['status'] == 'skipped']
    used_local = [r for r in successful if r.get('source_was_local', False)]

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total processed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"  - From local files: {len(used_local)}")
    print(f"  - From HuggingFace: {len(successful) - len(used_local)}")
    print(f"Skipped (already exists): {len(skipped)}")
    print(f"Output directory: {output_dir}")
    print("="*80)

    return 0


if __name__ == "__main__":
    exit(main())
