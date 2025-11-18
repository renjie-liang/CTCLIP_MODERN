"""
Data loading module for CT-CLIP

This module provides WebDataset-based data loading for preprocessed CT volumes.
All data is stored in preprocessed format (already rescaled, normalized, and resized).
"""

from .webdataset_loader import CTReportWebDataset

__all__ = ['CTReportWebDataset']
