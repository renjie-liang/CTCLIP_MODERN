"""
Configuration for NPZ conversion from HuggingFace nii.gz files.

All parameters for the conversion pipeline are defined here.
"""

# ============================================================================
# HuggingFace Repository
# ============================================================================
HF_REPO_ID = "ibrahimhamamci/CT-RATE"
HF_REPO_TYPE = "dataset"

# ============================================================================
# Output Paths
# ============================================================================
OUTPUT_BASE_DIR = "/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset"

# Split-specific output directories
OUTPUT_DIRS = {
    'train': f"{OUTPUT_BASE_DIR}/train_npz",
    'valid': f"{OUTPUT_BASE_DIR}/valid_npz"
}

# ============================================================================
# Preprocessing Parameters
# ============================================================================

# Target spacing (mm) - [z_spacing, y_spacing, x_spacing]
TARGET_SPACING = [1.5, 0.75, 0.75]

# Target shape after crop/pad - (D, H, W)
TARGET_SHAPE = (240, 480, 480)

# HU value clipping range
HU_CLIP_MIN = -1024  # Air
HU_CLIP_MAX = 3000   # High-density structures (preserve bone)

# Padding value (use air HU value)
PAD_VALUE = -1024

# Storage dtype
STORAGE_DTYPE = "int16"

# Target orientation (LPS = Left-Posterior-Superior)
TARGET_ORIENTATION = "LPS"

# ============================================================================
# Directory Structure
# ============================================================================

# Whether to organize NPZ files in subdirectories (patient/series structure)
# True:  output/train_001/train_001_a_1.npz
# False: output/train_001_a_1.npz
USE_NESTED_STRUCTURE = True

# ============================================================================
# Local Source Directories (Already Downloaded Files)
# ============================================================================

# Local directories containing already-downloaded nii.gz files
# These will be used first before downloading from HuggingFace
LOCAL_SOURCE_DIRS = {
    'train': "/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/train_fixed",
    'valid': "/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/valid_fixed"
}

# ============================================================================
# Download Settings
# ============================================================================

# Temporary directory for downloads from HuggingFace
# (Only used if file not found in LOCAL_SOURCE_DIRS)
TEMP_DIR = "/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/temp_downloads"

# Whether to delete source nii.gz files after successful conversion
# True: Delete after processing (to save space)
# False: Keep files
DELETE_SOURCE_AFTER_CONVERSION = False

# ============================================================================
# Processing Settings
# ============================================================================

# Maximum number of files to process (for testing)
# None = process all files
MAX_FILES = None  # Set to 20 for initial testing

# Random seed for sampling (if MAX_FILES is set)
RANDOM_SEED = 42

# Whether to skip existing NPZ files (resume capability)
SKIP_EXISTING = True

# ============================================================================
# Validation Settings
# ============================================================================

# Whether to validate orientation before processing
CHECK_ORIENTATION = True

# Whether to validate spacing after resampling
VALIDATE_SPACING = True

# Tolerance for spacing validation (mm)
SPACING_TOLERANCE = 0.01

# ============================================================================
# Logging
# ============================================================================

# Verbose output
VERBOSE = True

# Print progress every N files
PROGRESS_INTERVAL = 10

# ============================================================================
# Metadata Settings
# ============================================================================

# What information to store in NPZ metadata
METADATA_FIELDS = [
    'study_id',
    'source_file',
    'affine',
    'orientation',
    'spacing',
    'shape',
    'crop_bbox',
    'pad_params',
    'preprocessing',
    'quality'
]

# ============================================================================
# Split Configurations
# ============================================================================

SPLIT_CONFIGS = {
    'train': {
        'hf_path_pattern': 'dataset/train_fixed',
        'output_dir': OUTPUT_DIRS['train']
    },
    'valid': {
        'hf_path_pattern': 'dataset/valid_fixed',
        'output_dir': OUTPUT_DIRS['valid']
    }
}
