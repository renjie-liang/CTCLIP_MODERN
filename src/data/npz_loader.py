"""
NPZ-based CT Report Dataset for loading preprocessed volumes.

This dataset loads PREPROCESSED CT volumes from NPZ files.
All volumes are already:
- Resized to uniform spacing (0.75mm x 0.75mm x 1.5mm)
- Cropped/padded to (480, 480, 240)
- Saved as int16 for efficient storage

During loading, we only need to:
- Apply windowing (HU clipping)
- Normalize to [-1, 1] range
- Convert to tensor and permute dimensions
"""

import os
import glob
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class CTReportNPZDataset(Dataset):
    """
    NPZ-based CT Report Dataset.

    This class loads preprocessed CT volumes from NPZ files and associated
    reports/labels from CSV files.

    Args:
        data_folder (str): Path to folder containing NPZ files
        reports_file (str): Path to CSV file with reports (columns: VolumeName, Findings_EN, Impressions_EN)
        meta_file (str): Path to CSV file with metadata
        labels_file (str): Path to CSV file with labels
        min_hu (int): Minimum HU value for windowing (default: -1000)
        max_hu (int): Maximum HU value for windowing (default: 1000)
        mode (str): "train" or "val" - for logging purposes
    """

    def __init__(
        self,
        data_folder: str,
        reports_file: str,
        meta_file: str,
        labels_file: str,
        min_hu: int = -1000,
        max_hu: int = 1000,
        mode: str = "train"
    ):
        self.data_folder = data_folder
        self.min_hu = min_hu
        self.max_hu = max_hu
        self.mode = mode

        print(f"[{self.mode.upper()}] Initializing NPZ Dataset from: {data_folder}")
        print(f"[{self.mode.upper()}] Windowing: [{min_hu}, {max_hu}] HU")

        # Load metadata
        print(f"[{self.mode.upper()}] Loading metadata...")
        self.reports_dict = self._load_reports(reports_file)
        self.labels_dict = self._load_labels(labels_file)

        # Find NPZ files and filter by existence
        print(f"[{self.mode.upper()}] Scanning for NPZ files...")
        self.samples = self._prepare_samples()

        print(f"[{self.mode.upper()}] Dataset ready: {len(self.samples)} samples")

    def _load_reports(self, reports_file: str) -> dict:
        """Load report text from CSV file."""
        df = pd.read_csv(reports_file)
        reports_dict = {}

        for _, row in df.iterrows():
            # Extract study_id from VolumeName (remove .nii.gz extension)
            volume_name = str(row['VolumeName'])
            study_id = volume_name.replace('.nii.gz', '')

            # Combine findings and impressions
            findings = str(row.get('Findings_EN', ''))
            impressions = str(row.get('Impressions_EN', ''))

            # Skip "Not given." texts
            if findings == "Not given.":
                findings = ""
            if impressions == "Not given.":
                impressions = ""

            # Combine and clean
            report_text = f"{findings}\n{impressions}".strip()
            report_text = self._clean_text(report_text)

            reports_dict[study_id] = report_text

        print(f"  Loaded {len(reports_dict)} reports")
        return reports_dict

    def _load_labels(self, labels_file: str) -> dict:
        """Load labels from CSV file."""
        df = pd.read_csv(labels_file)

        # Get label columns (exclude VolumeName)
        label_cols = [col for col in df.columns if col != 'VolumeName']

        labels_dict = {}
        for _, row in df.iterrows():
            # Extract study_id from VolumeName
            volume_name = str(row['VolumeName'])
            study_id = volume_name.replace('.nii.gz', '')

            # Get one-hot labels
            labels = row[label_cols].values.astype(np.float32)
            labels_dict[study_id] = labels

        print(f"  Loaded {len(labels_dict)} label sets ({len(label_cols)} classes)")
        return labels_dict

    def _clean_text(self, text: str) -> str:
        """Clean report text by removing special characters."""
        text = str(text)
        if text == "Not given.":
            return ""
        text = text.replace('"', '')
        text = text.replace("'", '')
        text = text.replace('(', '')
        text = text.replace(')', '')
        return text.strip()

    def _prepare_samples(self) -> List[Tuple[str, str]]:
        """
        Find all NPZ files and match with metadata.
        Filters out files that don't exist or lack metadata.
        """
        # Find NPZ files (support multiple directory structures)
        npz_files_flat = glob.glob(os.path.join(self.data_folder, '*.npz'))
        npz_files_1layer = glob.glob(os.path.join(self.data_folder, '*', '*.npz'))
        npz_files_2layer = glob.glob(os.path.join(self.data_folder, '*', '*', '*.npz'))

        # Use the structure with most files
        all_structures = [
            (len(npz_files_flat), npz_files_flat, "flat"),
            (len(npz_files_1layer), npz_files_1layer, "1-layer"),
            (len(npz_files_2layer), npz_files_2layer, "2-layer")
        ]
        all_structures.sort(reverse=True, key=lambda x: x[0])

        num_found, npz_files, structure = all_structures[0]
        print(f"  Found {num_found} NPZ files ({structure} structure)")

        # Match with metadata and filter
        samples = []
        missing_reports = 0
        missing_labels = 0

        for npz_path in tqdm(npz_files, desc="Filtering samples"):
            # Extract study_id from filename
            study_id = Path(npz_path).stem

            # Check if file exists
            if not os.path.exists(npz_path):
                continue

            # Check if we have metadata
            has_report = study_id in self.reports_dict
            has_labels = study_id in self.labels_dict

            if not has_report:
                missing_reports += 1
            if not has_labels:
                missing_labels += 1

            # Only include if we have both report and labels
            if has_report and has_labels:
                samples.append((npz_path, study_id))

        print(f"  Matched {len(samples)} / {num_found} samples")
        if missing_reports > 0:
            print(f"  Skipped {missing_reports} samples without reports")
        if missing_labels > 0:
            print(f"  Skipped {missing_labels} samples without labels")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, np.ndarray, str]:
        """
        Load and process a single sample.

        Returns:
            volume_tensor: (1, D, H, W) = (1, 240, 480, 480) float32 in [-1, 1]
            report_text: str
            labels: (num_classes,) float32
            study_id: str
        """
        npz_path, study_id = self.samples[idx]

        # 1. Load NPZ data (int16, shape: 480, 480, 240)
        data = np.load(npz_path)
        volume = data['volume']  # int16, (H, W, D) = (480, 480, 240)

        # 2. Apply windowing (HU clipping)
        volume = np.clip(volume, self.min_hu, self.max_hu)

        # 3. Normalize to [-1, 1]
        volume = (volume - self.min_hu) / (self.max_hu - self.min_hu)  # [0, 1]
        volume = volume * 2.0 - 1.0  # [-1, 1]

        # 4. Convert to tensor
        volume_tensor = torch.from_numpy(volume).float()  # (480, 480, 240)

        # 5. Permute (H, W, D) -> (D, H, W)
        volume_tensor = volume_tensor.permute(2, 0, 1)  # (240, 480, 480)

        # 6. Add channel dimension
        volume_tensor = volume_tensor.unsqueeze(0)  # (1, 240, 480, 480)

        # 7. Get report text
        report_text = self.reports_dict[study_id]

        # 8. Get labels
        labels = self.labels_dict[study_id]

        return volume_tensor, report_text, labels, study_id


def create_npz_dataset_from_config(data_cfg: dict, mode: str = "train"):
    """
    Factory function to create NPZ Dataset from config.

    Args:
        data_cfg (dict): Data configuration with paths
        mode (str): "train" or "val"

    Returns:
        CTReportNPZDataset instance
    """
    # Map mode to config keys
    if mode == "train":
        data_folder = data_cfg['train_npz_dir']
        reports_file = data_cfg['reports_train']
        meta_file = data_cfg['train_meta']
        labels_file = data_cfg['labels_train']
    elif mode == "val":
        data_folder = data_cfg['val_npz_dir']
        reports_file = data_cfg['reports_val']
        meta_file = data_cfg['val_meta']
        labels_file = data_cfg['labels_val']
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'val'")

    dataset = CTReportNPZDataset(
        data_folder=data_folder,
        reports_file=reports_file,
        meta_file=meta_file,
        labels_file=labels_file,
        min_hu=data_cfg.get('min_hu', -1000),
        max_hu=data_cfg.get('max_hu', 1000),
        mode=mode
    )

    return dataset
