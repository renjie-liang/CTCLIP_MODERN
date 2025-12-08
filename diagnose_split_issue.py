#!/usr/bin/env python3
"""Diagnose split parameter issue"""

import argparse
import sys
sys.path.insert(0, '/home/user/CTCLIP_MODERN/scripts_important')

from config_npz_conversion import SPLIT_CONFIGS, LOCAL_SOURCE_DIRS

def main():
    """Simulate the script's parameter parsing"""
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

    # Parse the command line arguments
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"Argument Parsing Diagnostic")
    print(f"{'='*80}\n")

    print(f"Command line args: {sys.argv}")
    print(f"\nParsed arguments:")
    print(f"  args.split = '{args.split}'")
    print(f"  args.max_files = {args.max_files}")

    print(f"\n{'='*80}")
    print(f"Configuration for split='{args.split}'")
    print(f"{'='*80}\n")

    # Show what configuration will be used
    hf_path_pattern = SPLIT_CONFIGS[args.split]['hf_path_pattern']
    output_dir = SPLIT_CONFIGS[args.split]['output_dir']
    local_source_dir = LOCAL_SOURCE_DIRS[args.split]

    print(f"HF path pattern: '{hf_path_pattern}'")
    print(f"Output directory: '{output_dir}'")
    print(f"Local source directory: '{local_source_dir}'")

    print(f"\n{'='*80}")
    print(f"File Filtering Test")
    print(f"{'='*80}\n")

    # Simulate file filtering
    test_files = [
        'dataset/train_fixed/train_001/train_001_a/train_001_a_1.nii.gz',
        'dataset/train_fixed/train_002/train_002_a/train_002_a_1.nii.gz',
        'dataset/valid_fixed/valid_001/valid_001_a/valid_001_a_1.nii.gz',
        'dataset/valid_fixed/valid_002/valid_002_a/valid_002_a_1.nii.gz',
    ]

    print(f"Testing files:")
    for f in test_files:
        matches = f.startswith(hf_path_pattern) and f.endswith('.nii.gz')
        status = "✓ MATCH" if matches else "✗ no match"
        print(f"  {status}: {f}")

    matched_files = [
        f for f in test_files
        if f.startswith(hf_path_pattern) and f.endswith('.nii.gz')
    ]

    print(f"\nTotal matched: {len(matched_files)}")
    print(f"\n{'='*80}\n")

    if args.split == 'valid':
        if len(matched_files) == 2 and all('valid' in f for f in matched_files):
            print("✓ CORRECT: Split parameter is working correctly!")
            print("  When --split valid is used, only valid files are matched.")
        else:
            print("✗ ERROR: Split parameter is NOT working correctly!")
            print("  Expected to match only valid files, but got something else.")
    elif args.split == 'train':
        if len(matched_files) == 2 and all('train' in f for f in matched_files):
            print("✓ CORRECT: Split parameter is working correctly!")
            print("  When --split train is used, only train files are matched.")
        else:
            print("✗ ERROR: Split parameter is NOT working correctly!")
            print("  Expected to match only train files, but got something else.")

if __name__ == "__main__":
    main()
