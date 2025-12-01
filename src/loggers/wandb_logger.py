"""
WandB (Weights & Biases) logger implementation
"""

from typing import Dict, Any, Optional
from pathlib import Path
import warnings

from .base_logger import BaseLogger


class WandBLogger(BaseLogger):
    """
    WandB logger

    Usage example:
        logger = WandBLogger(
            config=config,
            project="ct-clip",
            entity="your-username"
        )

        logger.log_metrics({'loss': 0.5}, step=100, prefix='train')
        logger.finish()
    """

    def __init__(
        self,
        config: Dict[str, Any],
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[list] = None,
        group: Optional[str] = None,
        job_type: str = "train",
        mode: str = "online",
        resume_id: Optional[str] = None
    ):
        """
        Args:
            config: Complete configuration dictionary (will be logged as hyperparameters)
            project: WandB project name
            entity: WandB username/team name
            name: Run name
            tags: List of tags
            group: Experiment group name
            job_type: Job type
            mode: 'online', 'offline', or 'disabled'
            resume_id: WandB run id to resume from (for continuing interrupted runs)
        """
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            raise ImportError(
                "wandb is not installed. Install it with: pip install wandb"
            )

        # Extract experiment info from config
        exp_config = config.get('experiment', {})
        name = name or exp_config.get('name', 'unnamed-run')
        tags = tags or exp_config.get('tags', [])

        # Initialize wandb run
        if resume_id:
            # Resume existing run
            print(f"Resuming WandB run: {resume_id}")
            self.run = wandb.init(
                project=project,
                entity=entity,
                id=resume_id,
                resume="allow",  # Allow resuming if run exists, otherwise create new
                config=config,
                mode=mode
            )
            print(f"✓ WandB run resumed: {self.run.url}")
        else:
            # Create new run
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=name,
                tags=tags,
                group=group,
                job_type=job_type,
                config=config,
                mode=mode
            )
            print(f"WandB run initialized: {self.run.url}")

    def get_run_id(self) -> Optional[str]:
        """Get current WandB run ID for checkpointing"""
        if self.run is not None:
            return self.run.id
        return None

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = ""
    ) -> None:
        """
        Log metrics to WandB

        Args:
            metrics: {'loss': 0.5, 'auroc': 0.85}
            step: Step number
            prefix: Prefix such as 'train/', 'val/'
        """
        # Add prefix
        if prefix:
            if not prefix.endswith('/'):
                prefix = prefix + '/'
            formatted_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        else:
            formatted_metrics = metrics

        # Log to wandb
        self.run.log(formatted_metrics, step=step)

    def log_hyperparameters(self, config: Dict[str, Any]) -> None:
        """
        Update hyperparameters

        Args:
            config: Configuration dictionary
        """
        self.run.config.update(config, allow_val_change=True)

    def log_text(self, key: str, text: str, step: Optional[int] = None) -> None:
        """
        Log text

        Args:
            key: Text key name
            text: Text content
            step: Step number
        """
        self.run.log({key: self.wandb.Html(text)}, step=step)

    def log_artifact(self, file_path: str, artifact_type: str = "file") -> None:
        """
        Upload file to WandB

        Args:
            file_path: File path
            artifact_type: File type
        """
        artifact = self.wandb.Artifact(
            name=Path(file_path).stem,
            type=artifact_type
        )
        artifact.add_file(file_path)
        self.run.log_artifact(artifact)

    def watch_model(self, model, log_freq: int = 100) -> None:
        """
        Monitor model gradients and parameters

        Args:
            model: PyTorch model
            log_freq: Logging frequency
        """
        self.run.watch(model, log="all", log_freq=log_freq)

    def finish(self) -> None:
        """Finish WandB run"""
        if self.run is not None:
            self.run.finish()
            print("WandB run finished")
