"""
Data loading module for CT-CLIP

This module provides data loading for preprocessed CT volumes in two formats:
- WebDataset: TAR-based format for efficient sequential loading
- NPZ: Individual file format for flexibility

All data is stored in preprocessed format (already rescaled, normalized, and resized).
"""

from .webdataset_loader import CTReportWebDataset
from .npz_loader import CTReportNPZDataset

__all__ = ['CTReportWebDataset', 'CTReportNPZDataset']
