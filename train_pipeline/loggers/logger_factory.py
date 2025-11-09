"""
Logger工厂函数

根据配置自动创建合适的logger
"""

from typing import Dict, Any, List
import warnings

from .base_logger import BaseLogger
from .wandb_logger import WandBLogger
from .console_logger import ConsoleLogger


class MultiLogger(BaseLogger):
    """
    组合多个logger

    同时记录到多个后端
    """

    def __init__(self, loggers: List[BaseLogger]):
        """
        Args:
            loggers: logger列表
        """
        self.loggers = loggers

    def log_metrics(self, metrics, step=None, prefix=""):
        """记录到所有logger"""
        for logger in self.loggers:
            logger.log_metrics(metrics, step, prefix)

    def log_hyperparameters(self, config):
        """记录到所有logger"""
        for logger in self.loggers:
            logger.log_hyperparameters(config)

    def log_text(self, key, text, step=None):
        """记录到所有logger"""
        for logger in self.loggers:
            logger.log_text(key, text, step)

    def log_artifact(self, file_path, artifact_type="file"):
        """记录到所有logger"""
        for logger in self.loggers:
            logger.log_artifact(file_path, artifact_type)

    def watch_model(self, model, log_freq=100):
        """监控模型（只有支持的logger）"""
        for logger in self.loggers:
            try:
                logger.watch_model(model, log_freq)
            except:
                pass  # 某些logger可能不支持

    def finish(self):
        """结束所有logger"""
        for logger in self.loggers:
            logger.finish()


def create_logger(config: Dict[str, Any]) -> BaseLogger:
    """
    根据配置创建logger

    Args:
        config: 完整配置字典

    Returns:
        Logger实例（可能是MultiLogger）

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

    # TensorBoard Logger (可选，暂未实现)
    if logging_config.get('use_tensorboard', False):
        warnings.warn("TensorBoard logger not implemented yet")

    # 确保至少有一个logger
    if not loggers:
        warnings.warn("No logger enabled, using default console logger")
        loggers.append(ConsoleLogger())

    # 如果只有一个logger，直接返回
    if len(loggers) == 1:
        return loggers[0]

    # 多个logger，返回MultiLogger
    return MultiLogger(loggers)
