# NPZ Conversion from HuggingFace nii.gz Files

Convert CT-RATE nii.gz files from HuggingFace to NPZ format with complete preprocessing and metadata tracking.

## Quick Start

### Test on 20 validation samples

```bash
cd /path/to/CTCLIP_MODERN

python scripts/build_npz_from_hf.py \
    --split valid \
    --max-files 20 \
    --random-seed 42
```

### Full conversion

```bash
# Validation set
python scripts/build_npz_from_hf.py --split valid

# Training set
python scripts/build_npz_from_hf.py --split train
```

## Features

### 1. Use Local Files First
- Scans local directories for already-downloaded nii.gz files
- Only downloads from HuggingFace if not found locally
- Supports nested directory structure (patient/series/scan)

**Local directories:**
- Valid: `/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/valid_fixed`
- Train: `/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/train_fixed`

### 2. Complete Preprocessing Pipeline
- **Orientation**: Check and standardize to LPS
- **HU Clipping**: [-1024, 3000] (preserve bone, remove artifacts)
- **Resampling**: To [1.5, 0.75, 0.75] mm spacing
- **Crop/Pad**: To (240, 480, 480) shape (D, H, W)
- **Storage**: int16 format (HU values)

### 3. Metadata Tracking
Each NPZ file contains:
- **volume**: (480, 480, 240) int16 array
- **metadata**: Complete preprocessing information
  - Original and final affine matrices
  - Orientation code
  - Original and final shapes
  - Crop bounding box (for reversing)
  - Pad parameters (for reversing)
  - Preprocessing parameters
  - Quality metrics (HU range, etc.)

### 4. Source File Cleanup
- Downloaded files are always deleted after processing
- Local files can be optionally deleted (config: `DELETE_SOURCE_AFTER_CONVERSION`)

## Configuration

Edit `scripts/config_npz_conversion.py` to customize:

```python
# Preprocessing
TARGET_SPACING = [1.5, 0.75, 0.75]  # z, y, x in mm
TARGET_SHAPE = (240, 480, 480)      # D, H, W
HU_CLIP_MIN = -1024
HU_CLIP_MAX = 3000
PAD_VALUE = -1024

# Local source directories
LOCAL_SOURCE_DIRS = {
    'train': "/orange/.../train_fixed",
    'valid': "/orange/.../valid_fixed"
}

# Output directories
OUTPUT_DIRS = {
    'train': "/orange/.../train_npz",
    'valid': "/orange/.../valid_npz"
}

# Delete local files after conversion
DELETE_SOURCE_AFTER_CONVERSION = False  # Set to True to save space
```

## Output Structure

```
/orange/.../valid_npz/
├── valid_001/
│   ├── valid_001_a_1.npz
│   ├── valid_001_a_2.npz
│   └── valid_001_b_1.npz
├── valid_002/
│   └── ...
└── manifest.json
```

Each NPZ file is ~110 MB (int16 storage).

## Resume Support

The script automatically skips already-processed files:

```bash
# Run again - will skip existing NPZ files
python scripts/build_npz_from_hf.py --split valid
```

## Loading NPZ Files

```python
import numpy as np
import json

# Load NPZ
data = np.load('valid_001_a_1.npz')

# Get volume
volume = data['volume']  # (480, 480, 240) int16
print(f"Shape: {volume.shape}")
print(f"HU range: [{volume.min()}, {volume.max()}]")

# Get metadata
metadata = json.loads(data['metadata'].decode('utf-8'))
print(f"Study ID: {metadata['study_id']}")
print(f"Orientation: {metadata['orientation']}")
print(f"Final affine:\n{np.array(metadata['affine']['final'])}")
```

## Expected Output

```
================================================================================
Build NPZ Dataset from HuggingFace
================================================================================
Split: valid
Output dir: /orange/.../valid_npz
Temp dir: /orange/.../temp_downloads
Target spacing: [1.5, 0.75, 0.75]
Target shape: (240, 480, 480)
HU clip range: [-1024, 3000]
Delete source files: False

🔍 Scanning local directory: /orange/.../valid_fixed
   Found 498 local nii.gz files

📋 Listing files from ibrahimhamamci/CT-RATE (split=valid)...
   Found 498 valid files on HuggingFace

🔍 Scanning existing NPZ files in /orange/.../valid_npz...
   Found 0 existing NPZ files

📊 Status:
   Total files on HF: 498
   Already processed: 0
   Missing: 498

⚠️  Limiting to 20 files for testing

🚀 Processing 20 files...
   Files available locally: 20/20
   Files to download: 0/20

================================================================================
Processing: valid_001_a_1
================================================================================
   [1] Using local file...
      ✓ Local path: /orange/.../valid_fixed/valid_001/valid_001_a/valid_001_a_1.nii.gz
   [2] Loading NIfTI...
      Original shape: (512, 512, 350)
      Original spacing (x,y,z): (0.75, 0.75, 1.5)
   [3] Checking orientation...
      Current orientation: LPS
      ✓ Already in LPS orientation
   [4] Clipping HU values...
      Range before clip: [-1024.0, 2850.3]
      Range after clip: [-1024.0, 2850.3]
   [5] Resampling to target spacing...
      ...
   [9] Saving NPZ...
      ✓ Saved to: /orange/.../valid_npz/valid_001/valid_001_a_1.npz
      Size: 110.59 MB

...

================================================================================
SUMMARY
================================================================================
Total processed: 20
Successful: 20
  - From local files: 20
  - From HuggingFace: 0
Skipped (already exists): 0
Output directory: /orange/.../valid_npz
================================================================================
```

## Troubleshooting

### No local files found
- Check if `LOCAL_SOURCE_DIRS` path is correct
- Verify nii.gz files exist in subdirectories
- Script searches recursively, so nested structure is OK

### Out of memory
- Reduce `--max-files` for testing
- Processing is sequential (no parallelization)

### Orientation errors
- Most files should already be in LPS
- Script will auto-convert if needed
- Check output logs for orientation warnings
