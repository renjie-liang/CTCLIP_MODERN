#!/usr/bin/env python3
"""Quick script to check file formats on Hugging Face."""

from huggingface_hub import list_repo_files

repo_id = "ibrahimhamamci/CT-RATE"

print("Fetching file list from Hugging Face...")
files = list_repo_files(repo_id=repo_id, repo_type="dataset")

# Check train_fixed
train_files = [f for f in files if f.startswith("dataset/train_fixed/")]
val_files = [f for f in files if f.startswith("dataset/valid_fixed/")]

print(f"\nTrain files: {len(train_files)}")
if train_files:
    print("First 5 examples:")
    for f in train_files[:5]:
        print(f"  {f}")

print(f"\nValidation files: {len(val_files)}")
if val_files:
    print("First 5 examples:")
    for f in val_files[:5]:
        print(f"  {f}")

# Count file types
extensions = {}
for f in train_files + val_files:
    if f.endswith('.nii.gz'):
        ext = '.nii.gz'
    elif f.endswith('.npz'):
        ext = '.npz'
    elif f.endswith('.csv'):
        ext = '.csv'
    else:
        ext = 'other'
    extensions[ext] = extensions.get(ext, 0) + 1

print(f"\nFile types:")
for ext, count in sorted(extensions.items()):
    print(f"  {ext}: {count}")
