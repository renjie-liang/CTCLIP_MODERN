"""
WebDataset-based CT Report Dataset for efficient loading.

This dataset loads PREPROCESSED CT volumes from WebDataset TAR files.
All volumes are already:
- Resized to uniform spacing (0.75mm x 0.75mm x 1.5mm)
- Normalized to [-1, 1] range
- Cropped/padded to (480, 480, 240)
- Saved as float16 for efficient storage

No CPU preprocessing is performed during loading - data is ready to use.
"""

import json
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import webdataset as wds


class CTReportWebDataset:
    """
    WebDataset-based CT Report Dataset.

    This class wraps webdataset functionality to provide a PyTorch-compatible dataset
    for PREPROCESSED CT volumes and reports stored in WebDataset TAR format.

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
        print(f"[{self.mode.upper()}] ⚡ Using PREPROCESSED data (fast loading, no CPU preprocessing)")

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

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Note: WebDataset is an IterableDataset and doesn't naturally support len().
        We provide this for compatibility with code that expects it.
        """
        if self.num_samples is not None:
            return self.num_samples
        else:
            raise ValueError("Cannot determine dataset length: manifest.json not found")

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

    def _decode_sample(self, sample):
        """
        Decode a preprocessed sample from WebDataset (FAST - no CPU preprocessing).

        Preprocessed volumes are already:
        - Resized to uniform spacing
        - Normalized to [-1, 1] range
        - Cropped/padded to (480, 480, 240)
        - Saved as float16

        Only need to:
        1. Read volume data (~50ms)
        2. Convert to float32
        3. Permute (H,W,D) -> (D,H,W) (~0.01ms)
        4. Add channel dimension (~0.001ms)

        Total: ~50-100ms (vs ~1200ms for full preprocessing)
        """
        try:
            # Load metadata (simplified)
            metadata = json.loads(sample['json'].decode('utf-8'))
            study_id = metadata['study_id']

            # Load preprocessed volume - already in final shape (480, 480, 240) float16
            # Use .copy() to create writable array and avoid PyTorch warning
            volume_data = np.frombuffer(sample['bin'], dtype=np.float16).reshape(480, 480, 240).copy()

            # Convert to float32 tensor
            volume_tensor = torch.from_numpy(volume_data).float()  # (480, 480, 240)

            # Permute (H, W, D) -> (D, H, W)
            volume_tensor = volume_tensor.permute(2, 0, 1)  # (240, 480, 480)

            # Add channel dimension
            volume_tensor = volume_tensor.unsqueeze(0)  # (1, 240, 480, 480)

            # Load report text
            report_text = sample['txt'].decode('utf-8')
            report_text = self._clean_text(report_text)

            # Load labels (copy to avoid read-only warning)
            # Handle missing labels gracefully (some samples may not have labels)
            if 'labels' in sample:
                labels = np.frombuffer(sample['labels'], dtype=np.float32).copy()
            elif 'cls' in sample:  # Alternative key name
                labels = np.frombuffer(sample['cls'], dtype=np.float32).copy()
            else:
                # If no labels found, create empty array (will be filled from CSV later)
                labels = np.array([], dtype=np.float32)

            # No embedding in standard mode
            embed_tensor = torch.empty(0, dtype=volume_tensor.dtype)

            return volume_tensor, report_text, labels, study_id, embed_tensor

        except Exception as e:
            study_id = metadata.get('study_id', 'unknown') if 'metadata' in locals() else 'unknown'
            print(f"ERROR decoding preprocessed sample {study_id}: {e}")
            import traceback
            traceback.print_exc()
            raise

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
            .map(self._decode_sample)  # Fast decoding of preprocessed data
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
        Create a standard PyTorch DataLoader with WebDataset as IterableDataset.

        This provides better compatibility with existing training code.
        """
        from torch.utils.data import DataLoader

        # shardshuffle should be False or a positive integer (not True)
        shard_shuffle = 100 if self.shuffle else False

        dataset = (
            wds.WebDataset(self.shard_pattern, shardshuffle=shard_shuffle, empty_check=False)
            .shuffle(self.buffer_size if self.shuffle else 0)
            .map(self._decode_sample)  # Fast decoding of preprocessed data
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
