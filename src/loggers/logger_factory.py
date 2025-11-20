"""
Logger factory functions

Automatically create appropriate loggers based on configuration
"""

from typing import Dict, Any, List
import warnings

from .base_logger import BaseLogger
from .wandb_logger import WandBLogger
from .console_logger import ConsoleLogger


class MultiLogger(BaseLogger):
    """
    Combine multiple loggers

    Log to multiple backends simultaneously
    """

    def __init__(self, loggers: List[BaseLogger]):
        """
        Args:
            loggers: List of loggers
        """
        self.loggers = loggers

    def log_metrics(self, metrics, step=None, prefix=""):
        """Log to all loggers"""
        for logger in self.loggers:
            logger.log_metrics(metrics, step, prefix)

    def log_hyperparameters(self, config):
        """Log to all loggers"""
        for logger in self.loggers:
            logger.log_hyperparameters(config)

    def log_text(self, key, text, step=None):
        """Log to all loggers"""
        for logger in self.loggers:
            logger.log_text(key, text, step)

    def log_artifact(self, file_path, artifact_type="file"):
        """Log to all loggers"""
        for logger in self.loggers:
            logger.log_artifact(file_path, artifact_type)

    def watch_model(self, model, log_freq=100):
        """Monitor model (only supported loggers)"""
        for logger in self.loggers:
            try:
                logger.watch_model(model, log_freq)
            except:
                pass  # Some loggers may not support this

    def finish(self):
        """Finish all loggers"""
        for logger in self.loggers:
            logger.finish()


def create_logger(config: Dict[str, Any]) -> BaseLogger:
    """
    Create logger based on configuration

    Args:
        config: Complete configuration dictionary

    Returns:
        Logger instance (may be MultiLogger)

    Example:
        config = load_config("configs/base_config.yaml")
        logger = create_logger(config)

        logger.log_metrics({'loss': 0.5}, step=100, prefix='train')
    """
    logging_config = config.get('logging', {})
    loggers = []

    # Console Logger
    if logging_config.get('use_console', True):
        console_logger = ConsoleLogger(print_timestamp=True)
        loggers.append(console_logger)
        print("✓ Console logger enabled")

    # WandB Logger
    if logging_config.get('use_wandb', False):
        try:
            wandb_config = logging_config.get('wandb', {})
            wandb_logger = WandBLogger(
                config=config,
                project=wandb_config.get('project', 'ct-clip'),
                entity=wandb_config.get('entity'),
                group=wandb_config.get('group'),
                job_type=wandb_config.get('job_type', 'train'),
                mode=wandb_config.get('mode', 'online')
            )
            loggers.append(wandb_logger)
            print("✓ WandB logger enabled")
        except Exception as e:
            warnings.warn(f"Failed to initialize WandB logger: {e}")
            print("✗ WandB logger disabled")

    # TensorBoard Logger (optional, not yet implemented)
    if logging_config.get('use_tensorboard', False):
        warnings.warn("TensorBoard logger not implemented yet")

    # Ensure at least one logger exists
    if not loggers:
        warnings.warn("No logger enabled, using default console logger")
        loggers.append(ConsoleLogger())

    # If only one logger, return it directly
    if len(loggers) == 1:
        return loggers[0]

    # Multiple loggers, return MultiLogger
    return MultiLogger(loggers)
