"""
Checkpoint manager

Features:
1. Save complete training state (model, optimizer, scheduler, random seeds, etc.)
2. Automatically manage checkpoint count (keep only the latest N)
3. Save best checkpoint
4. Support full training resumption
"""

import os
import random
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import numpy as np
import warnings


class CheckpointManager:
    """
    Complete checkpoint manager

    Usage example:
        manager = CheckpointManager(
            save_dir="./saves",
            keep_last_n=3,
            save_best=True,
            best_metric="auroc",
            best_metric_mode="max"
        )

        # Save checkpoint
        manager.save_checkpoint(
            epoch=10,
            global_step=5000,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics={'auroc': 0.85, 'loss': 0.3},
            config=config
        )

        # Load checkpoint
        checkpoint = manager.load_checkpoint(
            "saves/checkpoint_epoch_10.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler
        )
    """

    def __init__(
        self,
        save_dir: str,
        keep_last_n: int = 3,
        save_best: bool = True,
        best_metric: str = "auroc",
        best_metric_mode: str = "max"
    ):
        """
        Args:
            save_dir: Checkpoint save directory
            keep_last_n: Keep the latest N checkpoints (0=keep all)
            save_best: Whether to save best checkpoint
            best_metric: Metric name to determine best
            best_metric_mode: 'max' or 'min'
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.best_metric = best_metric
        self.best_metric_mode = best_metric_mode

        # Initialize best value
        if best_metric_mode == 'max':
            self.best_value = -float('inf')
        elif best_metric_mode == 'min':
            self.best_value = float('inf')
        else:
            raise ValueError(f"best_metric_mode must be 'max' or 'min', got {best_metric_mode}")

        # Track checkpoint history
        self.checkpoint_history = []

    def save_checkpoint(
        self,
        epoch: int,
        global_step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        extra_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save complete checkpoint

        Args:
            epoch: Current epoch
            global_step: Global step number
            model: PyTorch model
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            metrics: Validation metrics
            config: Configuration dictionary
            extra_state: Additional state to save

        Returns:
            Path to saved file
        """
        metrics = metrics or {}
        config = config or {}
        extra_state = extra_state or {}

        # Build checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': config,
            'extra_state': extra_state
        }

        # Save scheduler state
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # Save random states (for full reproducibility)
        checkpoint['random_states'] = self._get_random_states()

        # Save latest checkpoint
        latest_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, latest_path)

        print(f"Saved checkpoint: {latest_path}")

        # Update history
        self.checkpoint_history.append({
            'path': latest_path,
            'epoch': epoch,
            'metrics': metrics
        })

        # Manage checkpoint count
        if self.keep_last_n > 0:
            self._cleanup_old_checkpoints()

        # Save best checkpoint
        if self.save_best:
            self._save_best_if_needed(checkpoint, metrics, epoch)

        # Always keep a latest link
        self._update_latest_symlink(latest_path)

        return str(latest_path)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        load_random_states: bool = True,
        strict: bool = False  # Changed to False to support loading old checkpoints
    ) -> Dict[str, Any]:
        """
        Load checkpoint and restore training state

        Args:
            checkpoint_path: Checkpoint file path (or 'latest', 'best')
            model: PyTorch model
            optimizer: Optimizer (None=don't restore optimizer state)
            scheduler: Scheduler (None=don't restore scheduler state)
            load_random_states: Whether to restore random states
            strict: Whether to strictly match model state dict (False=allow partial loading)

        Returns:
            Checkpoint dictionary containing epoch, global_step, metrics, etc.
        """
        # Handle special paths
        if checkpoint_path == 'latest':
            checkpoint_path = self.save_dir / "latest.pt"
        elif checkpoint_path == 'best':
            checkpoint_path = self.save_dir / "best.pt"
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint from: {checkpoint_path}")

        # Load checkpoint (PyTorch 2.6 compatibility)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Restore model
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        print(f"✓ Model state restored")

        # Restore optimizer
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"✓ Optimizer state restored")

        # Restore scheduler
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"✓ Scheduler state restored")

        # Restore random states
        if load_random_states and 'random_states' in checkpoint:
            self._set_random_states(checkpoint['random_states'])
            print(f"✓ Random states restored")

        # Print restore info
        print(f"Resumed from epoch {checkpoint['epoch']}, step {checkpoint['global_step']}")
        if 'metrics' in checkpoint:
            print(f"Checkpoint metrics: {checkpoint['metrics']}")

        return checkpoint

    def _save_best_if_needed(
        self,
        checkpoint: Dict[str, Any],
        metrics: Dict[str, float],
        epoch: int
    ) -> None:
        """
        Save as best checkpoint if metrics are better

        Args:
            checkpoint: Checkpoint dictionary
            metrics: Validation metrics
            epoch: Current epoch
        """
        # Extract target metric
        # Support 'auroc' or 'macro_auroc'
        metric_value = None
        for key in [self.best_metric, f'macro_{self.best_metric}']:
            if key in metrics:
                metric_value = metrics[key]
                break

        if metric_value is None:
            warnings.warn(f"Metric '{self.best_metric}' not found in metrics, skipping best checkpoint")
            return

        # Determine if better
        is_better = False
        if self.best_metric_mode == 'max':
            is_better = metric_value > self.best_value
        else:
            is_better = metric_value < self.best_value

        if is_better:
            self.best_value = metric_value
            best_path = self.save_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"★ New best {self.best_metric}: {metric_value:.4f} (epoch {epoch}) - saved to {best_path}")

    def _cleanup_old_checkpoints(self) -> None:
        """Delete old checkpoints, keep only the latest N"""
        if len(self.checkpoint_history) <= self.keep_last_n:
            return

        # Sort by epoch
        sorted_history = sorted(self.checkpoint_history, key=lambda x: x['epoch'])

        # Delete oldest
        to_delete = sorted_history[:-self.keep_last_n]

        for item in to_delete:
            path = item['path']
            if path.exists() and path.name != 'best.pt' and path.name != 'latest.pt':
                path.unlink()
                print(f"Deleted old checkpoint: {path}")

        # Update history
        self.checkpoint_history = sorted_history[-self.keep_last_n:]

    def _update_latest_symlink(self, latest_path: Path) -> None:
        """Create or update latest symlink"""
        symlink_path = self.save_dir / "latest.pt"

        # Delete old symlink
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()

        # Create new symlink (if system supports it)
        try:
            symlink_path.symlink_to(latest_path.name)
        except OSError:
            # Symlinks not supported (e.g., Windows), copy instead
            import shutil
            shutil.copy(latest_path, symlink_path)

    def _get_random_states(self) -> Dict[str, Any]:
        """Get state of all random number generators"""
        states = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
        }

        # CUDA random states
        if torch.cuda.is_available():
            states['cuda'] = torch.cuda.get_rng_state_all()

        return states

    def _set_random_states(self, states: Dict[str, Any]) -> None:
        """Restore state of all random number generators"""
        random.setstate(states['python'])
        np.random.set_state(states['numpy'])
        torch.set_rng_state(states['torch'])

        if torch.cuda.is_available() and 'cuda' in states:
            torch.cuda.set_rng_state_all(states['cuda'])

    def list_checkpoints(self) -> list:
        """List all available checkpoints"""
        checkpoints = sorted(self.save_dir.glob("checkpoint_epoch_*.pt"))
        return [str(p) for p in checkpoints]

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get latest checkpoint path"""
        latest_path = self.save_dir / "latest.pt"
        if latest_path.exists():
            return str(latest_path)
        return None

    def get_best_checkpoint(self) -> Optional[str]:
        """Get best checkpoint path"""
        best_path = self.save_dir / "best.pt"
        if best_path.exists():
            return str(best_path)
        return None
