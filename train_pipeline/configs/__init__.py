"""
Configuration management for CT-CLIP training pipeline.
"""

from .config_loader import load_config, merge_configs

__all__ = ['load_config', 'merge_configs']
