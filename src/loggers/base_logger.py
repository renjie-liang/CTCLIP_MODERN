"""
Abstract logging interface

Defines methods that all loggers must implement
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path


class BaseLogger(ABC):
    """Base class for loggers"""

    @abstractmethod
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = ""
    ) -> None:
        """
        Log metrics

        Args:
            metrics: Metrics dictionary {'loss': 0.5, 'auroc': 0.85}
            step: Current step/epoch
            prefix: Prefix such as 'train/', 'val/'
        """
        pass

    @abstractmethod
    def log_hyperparameters(self, config: Dict[str, Any]) -> None:
        """
        Log hyperparameters

        Args:
            config: Configuration dictionary
        """
        pass

    @abstractmethod
    def log_text(self, key: str, text: str, step: Optional[int] = None) -> None:
        """
        Log text

        Args:
            key: Text key name
            text: Text content
            step: Step number
        """
        pass

    def log_artifact(self, file_path: str, artifact_type: str = "file") -> None:
        """
        Upload file

        Args:
            file_path: File path
            artifact_type: File type label
        """
        # Default implementation: do nothing
        pass

    def watch_model(self, model, log_freq: int = 100) -> None:
        """
        Monitor model (gradients, parameters, etc.)

        Args:
            model: PyTorch model
            log_freq: Logging frequency
        """
        # Default implementation: do nothing
        pass

    @abstractmethod
    def finish(self) -> None:
        """Finish logging"""
        pass

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.finish()
