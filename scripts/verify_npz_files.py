"""
Verify NPZ files structure without requiring PyTorch.

Usage:
    python scripts/verify_npz_files.py
"""

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path


def check_npz_directory():
    """Check if NPZ directory exists and contains files."""
    npz_dir = "/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/vaild_npz"

    print("="*80)
    print("NPZ Files Verification")
    print("="*80)
    print(f"\nChecking directory: {npz_dir}")

    if not os.path.exists(npz_dir):
        print(f"❌ Directory does not exist: {npz_dir}")
        return False

    print(f"✓ Directory exists")

    # Find NPZ files
    print("\nScanning for NPZ files...")
    npz_files = glob.glob(os.path.join(npz_dir, "*.npz"))

    if not npz_files:
        print("❌ No NPZ files found in directory")
        return False

    print(f"✓ Found {len(npz_files)} NPZ files")

    # Check first file
    print(f"\nInspecting first file: {Path(npz_files[0]).name}")
    try:
        data = np.load(npz_files[0])
        print(f"✓ File loaded successfully")
        print(f"  Keys in NPZ file: {list(data.keys())}")

        if 'volume' in data:
            volume = data['volume']
            print(f"  Volume shape: {volume.shape}")
            print(f"  Volume dtype: {volume.dtype}")
            print(f"  Volume min: {volume.min()}")
            print(f"  Volume max: {volume.max()}")
        else:
            print(f"  ⚠️  'volume' key not found!")

    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return False

    return True


def check_csv_files():
    """Check if CSV files exist."""
    print("\n" + "="*80)
    print("CSV Files Verification")
    print("="*80)

    csv_files = {
        'Reports': "/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/radiology_text_reports/validation_reports.csv",
        'Metadata': "/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/metadata/validation_metadata.csv",
        'Labels': "/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/multi_abnormality_labels/valid_predicted_labels.csv"
    }

    all_exist = True
    for name, path in csv_files.items():
        print(f"\n{name}: {path}")
        if os.path.exists(path):
            print(f"  ✓ File exists")
            try:
                df = pd.read_csv(path)
                print(f"  ✓ Loaded successfully: {len(df)} rows, {len(df.columns)} columns")
                print(f"  Columns: {list(df.columns)[:5]}...")
            except Exception as e:
                print(f"  ❌ Error loading CSV: {e}")
                all_exist = False
        else:
            print(f"  ❌ File does not exist")
            all_exist = False

    return all_exist


def main():
    """Run all verification checks."""
    print("\n" + "="*80)
    print("Data Files Verification for NPZ Loader")
    print("="*80)

    npz_ok = check_npz_directory()
    csv_ok = check_csv_files()

    print("\n" + "="*80)
    print("Summary")
    print("="*80)

    if npz_ok and csv_ok:
        print("\n✓ All files verified successfully!")
        print("  NPZ loader should work correctly.")
    else:
        print("\n❌ Some files are missing or have issues:")
        if not npz_ok:
            print("  - NPZ files: FAILED")
        if not csv_ok:
            print("  - CSV files: FAILED")

    print("="*80)


if __name__ == '__main__':
    main()
