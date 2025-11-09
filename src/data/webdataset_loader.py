"""
WebDataset-based CT Report Dataset for efficient loading.

This provides a drop-in replacement for CTReportDataset that uses WebDataset format
for significantly faster I/O and reduced storage requirements.
"""

import os
import json
import io
from pathlib import Path
from typing import Optional, Callable
import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds


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
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array


class CTReportWebDataset:
    """
    WebDataset-based CT Report Dataset.

    This class wraps webdataset functionality to provide a PyTorch-compatible dataset
    for CT volumes and reports stored in WebDataset TAR format.

    Args:
        shard_pattern (str): Pattern for shard files (e.g., "/path/to/shards/shard-{000000..000099}.tar")
        shuffle (bool): Whether to shuffle data (default: True for training)
        buffer_size (int): Shuffle buffer size (default: 1000)
        use_embedding (bool): If True, load precomputed embeddings instead of volumes
        mode (str): "train" or "val" - for logging purposes
    """

    def __init__(
        self,
        shard_pattern: str,
        shuffle: bool = True,
        buffer_size: int = 1000,
        use_embedding: bool = False,
        mode: str = "train"
    ):
        self.shard_pattern = shard_pattern
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.use_embedding = use_embedding
        self.mode = mode

        print(f"[{self.mode.upper()}] Initializing WebDataset from: {shard_pattern}")

        # Verify shards exist
        shard_dir = Path(shard_pattern).parent
        if not shard_dir.exists():
            raise ValueError(f"Shard directory does not exist: {shard_dir}")

        # Load manifest if available
        manifest_path = shard_dir / 'manifest.json'
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                self.manifest = json.load(f)
            self.num_samples = self.manifest['total_samples']
            print(f"[{self.mode.upper()}] Loaded manifest: {self.num_samples} samples")
        else:
            self.manifest = None
            self.num_samples = None
            print(f"[{self.mode.upper()}] Warning: No manifest.json found")

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

    def _process_volume(self, volume_data: np.ndarray, metadata: dict) -> torch.Tensor:
        """
        Process raw volume data (float16) to final tensor format.

        This replicates the logic from CTReportDataset._load_volume_from_npz

        Args:
            volume_data (np.ndarray): Raw volume data (float16)
            metadata (dict): Metadata including spacing information

        Returns:
            torch.Tensor: Processed volume tensor of shape (1, D, H, W)
        """
        # Convert float16 back to float32 for processing
        img_data = volume_data.astype(np.float32)

        # Get metadata
        slope = float(metadata["RescaleSlope"])
        intercept = float(metadata["RescaleIntercept"])

        # Parse XYSpacing (format: "[0.75, 0.75]")
        xy_spacing_str = str(metadata["XYSpacing"])
        xy_spacing = float(xy_spacing_str.strip("[]").split(",")[0])
        z_spacing = float(metadata["ZSpacing"])

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

    def _decode_sample(self, sample):
        """
        Decode a single sample from WebDataset.

        WebDataset provides samples as dicts with keys like 'bin', 'json', 'txt', 'labels'.
        """
        # Load metadata first (contains shape info for volume reconstruction)
        metadata = json.loads(sample['json'].decode('utf-8'))
        study_id = metadata['study_id']

        # Load volume data (stored as raw bytes, reconstruct from shape and dtype)
        volume_shape = tuple(metadata['volume_shape'])
        volume_dtype = np.dtype(metadata['volume_dtype'])
        volume_data = np.frombuffer(sample['bin'], dtype=volume_dtype).reshape(volume_shape)

        # Load report text
        report_text = sample['txt'].decode('utf-8')
        report_text = self._clean_text(report_text)

        # Load labels (binary data, decode as float32 array)
        labels = np.frombuffer(sample['labels'], dtype=np.float32)

        # Process volume
        if self.use_embedding:
            # For embedding mode, volume_data is already the embedding
            embed_tensor = torch.from_numpy(volume_data.astype(np.float32))
            volume_tensor = torch.empty(0, dtype=embed_tensor.dtype)
        else:
            # Process volume from float16 to final format
            volume_tensor = self._process_volume(volume_data, metadata)
            embed_tensor = torch.empty(0, dtype=volume_tensor.dtype)

        return volume_tensor, report_text, labels, study_id, embed_tensor

    def make_loader(self, batch_size: int, num_workers: int = 4, prefetch_factor: int = 2):
        """
        Create a WebDataset DataLoader.

        Args:
            batch_size (int): Batch size
            num_workers (int): Number of worker processes
            prefetch_factor (int): Number of batches to prefetch per worker

        Returns:
            DataLoader compatible object
        """
        dataset = (
            wds.WebDataset(self.shard_pattern, shardshuffle=self.shuffle, empty_check=False)
            .shuffle(self.buffer_size if self.shuffle else 0)
            # Don't use .decode() - we handle all decoding manually in _decode_sample
            .map(self._decode_sample)  # Custom decoding and processing
            .batched(batch_size)  # Create batches
        )

        # WebDataset creates its own DataLoader-like interface
        loader = wds.WebLoader(
            dataset,
            batch_size=None,  # Batching is done in the dataset pipeline
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=True,
            pin_memory=True
        )

        # For compatibility with PyTorch DataLoader interface
        if self.shuffle:
            loader = loader.unbatched().shuffle(1000).batched(batch_size)

        return loader

    def create_pytorch_dataloader(self, batch_size: int, num_workers: int = 4, prefetch_factor: int = 2):
        """
        Alternative: Create a standard PyTorch DataLoader with WebDataset as IterableDataset.

        This provides better compatibility with existing training code.
        """
        from torch.utils.data import DataLoader

        # shardshuffle should be False or a positive integer (not True)
        shard_shuffle = 100 if self.shuffle else False

        dataset = (
            wds.WebDataset(self.shard_pattern, shardshuffle=shard_shuffle, empty_check=False)
            .shuffle(self.buffer_size if self.shuffle else 0)
            # Don't use .decode() - we handle all decoding manually in _decode_sample
            .map(self._decode_sample)
        )

        # Build DataLoader kwargs
        loader_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'persistent_workers': True if num_workers > 0 else False,
            'pin_memory': True,
            'collate_fn': self._collate_fn
        }

        # Only add prefetch_factor if num_workers > 0
        if num_workers > 0:
            loader_kwargs['prefetch_factor'] = prefetch_factor

        loader = DataLoader(**loader_kwargs)

        return loader

    @staticmethod
    def _collate_fn(batch):
        """
        Custom collate function to handle batching.

        WebDataset returns tuples, we need to properly batch them.
        """
        volumes, reports, labels, study_ids, embeddings = zip(*batch)

        # Stack volumes (if not empty)
        if volumes[0].numel() > 0:
            volumes = torch.stack(volumes)
        else:
            volumes = torch.empty(0)

        # Stack embeddings (if not empty)
        if embeddings[0].numel() > 0:
            embeddings = torch.stack(embeddings)
        else:
            embeddings = torch.empty(0)

        # Stack labels
        labels = np.stack(labels)

        # Keep reports and study_ids as lists
        return volumes, list(reports), labels, list(study_ids), embeddings


def create_webdataset_from_config(data_cfg: dict, mode: str = "train"):
    """
    Factory function to create WebDataset from config.

    Args:
        data_cfg (dict): Data configuration with 'webdataset_shards_train' or 'webdataset_shards_val'
        mode (str): "train" or "val"

    Returns:
        CTReportWebDataset instance
    """
    shard_key = f'webdataset_shards_{mode}'

    if shard_key not in data_cfg:
        raise ValueError(f"Config missing '{shard_key}' key")

    dataset = CTReportWebDataset(
        shard_pattern=data_cfg[shard_key],
        shuffle=(mode == 'train'),
        buffer_size=data_cfg.get('shuffle_buffer_size', 1000),
        use_embedding=data_cfg.get('use_embedding', False),
        mode=mode
    )

    return dataset
