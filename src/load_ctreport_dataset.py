import os
import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import tqdm
from pathlib import Path


def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing.

    Args:
        array (torch.Tensor): Input array to be resized.
        current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
        target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
        np.ndarray: Resized array.
    """
    # Calculate new dimensions
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    # Resize the array
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array


class CTReportDataset(Dataset):
    """
    Unified CT Report Dataset for both training and inference.
    
    Loads preprocessed .npz files (faster than .nii.gz).
    Uses study_id (filename without extension) as the primary key for indexing.
    
    Args:
        data_folder (str): Directory containing .npz files
        reports_file (str): CSV file with reports (must have 'VolumeName' column)
        meta_file (str): CSV file with metadata (must have 'VolumeName' column)
        labels (str): CSV file with disease labels (must have 'VolumeName' column)
        embedding_dir (str): Directory for cached embeddings
        use_embedding (bool): If True, load precomputed embeddings instead of volumes
        mode (str): "train" or "val" - for logging purposes
    """
    
    def __init__(
        self, 
        data_folder: str, 
        reports_file: str, 
        meta_file: str, 
        labels: str, 
        embedding_dir: str = "", 
        use_embedding: bool = False,
        mode: str = "train"
    ):
        self.data_folder = data_folder
        self.embedding_dir = Path(embedding_dir) if embedding_dir else None
        self.use_embedding = use_embedding
        self.mode = mode
        
        # Load and index all data by study_id
        self.reports_dict = self._load_reports(reports_file)
        self.meta_df = self._load_metadata(meta_file)
        self.labels_dict = self._load_labels(labels)
        
        # Scan and prepare samples
        self.samples = self._prepare_samples()
        
        print(f"[{self.mode.upper()}] Dataset initialized: {len(self.samples)} samples")
        
    def _load_reports(self, reports_file: str) -> dict:
        """
        Load report texts and index by study_id.
        
        Returns:
            dict: {study_id: (findings, impressions)}
        """
        df = pd.read_csv(reports_file)
        reports_dict = {}
        
        for _, row in df.iterrows():
            findings = row.get("Findings_EN", "")
            impressions = row.get("Impressions_EN", "")
            
            # Extract study_id (remove .nii.gz extension if present)
            study_id = str(row['VolumeName']).replace(".nii.gz", "")
            reports_dict[study_id] = (findings, impressions)
        
        print(f"[{self.mode.upper()}] Loaded {len(reports_dict)} reports")
        return reports_dict

    def _load_metadata(self, meta_file: str) -> pd.DataFrame:
        """
        Load metadata and create study_id index.
        
        Returns:
            pd.DataFrame: Metadata with 'study_id' column
        """
        df = pd.read_csv(meta_file)
        df['study_id'] = df['VolumeName'].str.replace(".nii.gz", "", regex=False)
        df = df.set_index('study_id')
        
        print(f"[{self.mode.upper()}] Loaded metadata for {len(df)} studies")
        return df

    def _load_labels(self, labels_file: str) -> dict:
        """
        Load disease labels and index by study_id.
        
        Returns:
            dict: {study_id: np.array of shape (n_classes,)}
        """
        df = pd.read_csv(labels_file)
        df['study_id'] = df['VolumeName'].str.replace(".nii.gz", "", regex=False)
        
        # Get label columns (exclude VolumeName and study_id)
        label_cols = [col for col in df.columns if col not in ['VolumeName', 'study_id']]
        
        labels_dict = {}
        for _, row in df.iterrows():
            study_id = row['study_id']
            # Extract label values as float32 numpy array
            labels = row[label_cols].values.astype(np.float32)
            labels_dict[study_id] = labels
        
        print(f"[{self.mode.upper()}] Loaded labels for {len(labels_dict)} studies ({len(label_cols)} classes)")
        return labels_dict

    def _prepare_samples(self):
        """
        Scan data_folder for .npz files and match with reports/labels.

        Flexibly handles multiple directory structures:
        - 3-layer: data_folder/patient_id/accession_id/*.npz
        - 2-layer: data_folder/patient_id/*.npz
        - 1-layer: data_folder/*.npz

        Returns:
            List of tuples: (npz_file_path, study_id)
        """
        samples = []

        # Try to find .npz files in different structures
        print(f"[{self.mode.upper()}] Scanning data folder: {self.data_folder}")

        # Structure 1: Try 3-layer (patient/accession/*.npz)
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))
        patient_folders = [f for f in patient_folders if os.path.isdir(f)]

        npz_files_3layer = []
        npz_files_2layer = []

        for patient_folder in patient_folders:
            # Check for 2-layer structure first (patient/*.npz)
            npz_in_patient = glob.glob(os.path.join(patient_folder, '*.npz'))
            if npz_in_patient:
                npz_files_2layer.extend(npz_in_patient)

            # Check for 3-layer structure (patient/accession/*.npz)
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))
            accession_folders = [f for f in accession_folders if os.path.isdir(f)]

            for accession_folder in accession_folders:
                npz_in_accession = glob.glob(os.path.join(accession_folder, '*.npz'))
                if npz_in_accession:
                    npz_files_3layer.extend(npz_in_accession)

        # Structure 2: Try 1-layer (data_folder/*.npz)
        npz_files_1layer = glob.glob(os.path.join(self.data_folder, '*.npz'))

        # Determine which structure to use (pick the one with most files)
        structure_options = [
            (len(npz_files_3layer), "3-layer (patient/accession/*.npz)", npz_files_3layer),
            (len(npz_files_2layer), "2-layer (patient/*.npz)", npz_files_2layer),
            (len(npz_files_1layer), "1-layer (*.npz)", npz_files_1layer)
        ]
        structure_options.sort(reverse=True, key=lambda x: x[0])

        num_files, structure_name, npz_files = structure_options[0]

        if num_files == 0:
            print(f"[{self.mode.upper()}] ⚠️ WARNING: No .npz files found in {self.data_folder}")
            print(f"[{self.mode.upper()}] Checked:")
            print(f"  - 3-layer: {len(npz_files_3layer)} files")
            print(f"  - 2-layer: {len(npz_files_2layer)} files")
            print(f"  - 1-layer: {len(npz_files_1layer)} files")
            return []

        print(f"[{self.mode.upper()}] Detected {structure_name} - found {num_files} .npz files")

        # Match with reports/labels/metadata
        matched = 0
        missing_reports = 0
        missing_labels = 0
        missing_meta = 0

        for npz_file in tqdm.tqdm(npz_files, desc=f"Matching {self.mode} data"):
            study_id = Path(npz_file).stem  # Filename without extension

            # Check if we have all required data for this study
            if study_id not in self.reports_dict:
                missing_reports += 1
                continue
            if study_id not in self.labels_dict:
                missing_labels += 1
                continue
            if study_id not in self.meta_df.index:
                missing_meta += 1
                continue

            samples.append((npz_file, study_id))
            matched += 1

        # Print matching statistics
        print(f"[{self.mode.upper()}] Matching statistics:")
        print(f"  - Found {num_files} .npz files")
        print(f"  - Matched: {matched} samples")
        if missing_reports > 0:
            print(f"  - Missing reports: {missing_reports}")
        if missing_labels > 0:
            print(f"  - Missing labels: {missing_labels}")
        if missing_meta > 0:
            print(f"  - Missing metadata: {missing_meta}")

        return samples

    def __len__(self):
        return len(self.samples)

    def _load_volume_from_npz(self, npz_path: Path, study_id: str) -> torch.Tensor:
        """
        Load and preprocess volume from .npz file.
        
        Args:
            npz_path (Path): Path to .npz file
            study_id (str): Study identifier
            
        Returns:
            torch.Tensor: Preprocessed volume tensor of shape (1, D, H, W)
        """
        # Load data from npz
        img_data = np.load(npz_path, mmap_mode=None)["data"]
        # Get metadata for this study
        meta_row = self.meta_df.loc[study_id]
        slope = float(meta_row["RescaleSlope"])
        intercept = float(meta_row["RescaleIntercept"])
        
        # Parse XYSpacing (format: "[0.75, 0.75]")
        xy_spacing_str = str(meta_row["XYSpacing"])
        xy_spacing = float(xy_spacing_str.strip("[]").split(",")[0])
        z_spacing = float(meta_row["ZSpacing"])

        # Define target spacing
        target_x_spacing = 0.75
        target_y_spacing = 0.75
        target_z_spacing = 1.5

        current_spacing = (z_spacing, xy_spacing, xy_spacing)
        target_spacing = (target_z_spacing, target_x_spacing, target_y_spacing)

        # Apply rescale slope and intercept
        img_data = slope * img_data + intercept
        
        # Clip to HU range
        hu_min, hu_max = -1000, 1000
        img_data = np.clip(img_data, hu_min, hu_max)

        # Transpose to (D, H, W)
        img_data = img_data.transpose(2, 0, 1)

        # Convert to tensor and add batch/channel dims
        tensor = torch.tensor(img_data, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, D, H, W)

        # Resize to target spacing
        img_data = resize_array(tensor, current_spacing, target_spacing)
        img_data = img_data[0][0]  # Remove batch and channel dims
        img_data = np.transpose(img_data, (1, 2, 0))  # (H, W, D)

        # Normalize to [-1, 1] range
        img_data = (img_data / 1000).astype(np.float32)

        tensor = torch.tensor(img_data)
        
        # Crop/pad to target shape (480, 480, 240)
        target_shape = (480, 480, 240)
        h, w, d = tensor.shape
        dh, dw, dd = target_shape
        
        # Calculate crop/pad indices
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # Crop
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        # Pad if necessary
        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before
        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before
        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(
            tensor, 
            (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), 
            value=-1
        )

        # Permute to (D, H, W) and add channel dim
        tensor = tensor.permute(2, 0, 1)  # (D, H, W)
        tensor = tensor.unsqueeze(0)      # (1, D, H, W)

        return tensor

    def _clean_text(self, text: str) -> str:
        """Clean report text by removing special characters."""
        text = str(text)
        if text == "Not given.":
            return ""
        text = text.replace('"', '')
        text = text.replace("'", '')
        text = text.replace('(', '')
        text = text.replace(')', '')
        return text

    def __getitem__(self, index):
        """
        Get a sample from the dataset.
        
        Returns:
            Tuple of:
                - volume_tensor (torch.Tensor): Volume data or empty tensor if using embeddings
                - report_text (str): Cleaned report text
                - disease_labels (np.ndarray): One-hot encoded disease labels of shape (n_classes,)
                - study_id (str): Study identifier
                - embed_tensor (torch.Tensor): Precomputed embedding or empty tensor
        """
        npz_path, study_id = self.samples[index]
        npz_path = Path(npz_path)

        # Get report text
        findings, impressions = self.reports_dict[study_id]
        report_text = self._clean_text(findings) + self._clean_text(impressions)

        # Get disease labels
        disease_labels = self.labels_dict[study_id]  # Already numpy array
        
        # Load volume or embedding
        if self.use_embedding:
            # Load precomputed embedding
            embed_path = self.embedding_dir / f"{study_id}.npz"
            arr = np.load(embed_path, mmap_mode='r')["arr"]
            embed_tensor = torch.from_numpy(np.array(arr))  # Ensure it's a copy
            volume_tensor = torch.empty(0, dtype=embed_tensor.dtype)

        else:
            # Load volume from .npz
            volume_tensor = self._load_volume_from_npz(npz_path, study_id)
            embed_tensor = torch.empty(0, dtype=volume_tensor.dtype)
        
        return volume_tensor, report_text, disease_labels, study_id, embed_tensor