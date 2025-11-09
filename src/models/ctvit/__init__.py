"""
CTViT: 3D Vision Transformer for CT Volume Processing

A VQ-VAE autoencoder with factorized spatial-temporal attention
for medical CT volume encoding and reconstruction.
"""

from .ctvit import CTViT

__all__ = ['CTViT']
__version__ = '2.0.0'  # 重构版本
