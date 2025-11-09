"""
抽象日志接口

定义所有logger必须实现的方法
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path


class BaseLogger(ABC):
    """日志记录器基类"""

    @abstractmethod
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = ""
    ) -> None:
        """
        记录指标

        Args:
            metrics: 指标字典 {'loss': 0.5, 'auroc': 0.85}
            step: 当前步数/epoch
            prefix: 前缀，如 'train/', 'val/'
        """
        pass

    @abstractmethod
    def log_hyperparameters(self, config: Dict[str, Any]) -> None:
        """
        记录超参数

        Args:
            config: 配置字典
        """
        pass

    @abstractmethod
    def log_text(self, key: str, text: str, step: Optional[int] = None) -> None:
        """
        记录文本

        Args:
            key: 文本键名
            text: 文本内容
            step: 步数
        """
        pass

    def log_artifact(self, file_path: str, artifact_type: str = "file") -> None:
        """
        上传文件

        Args:
            file_path: 文件路径
            artifact_type: 文件类型标签
        """
        # 默认实现：不做任何事
        pass

    def watch_model(self, model, log_freq: int = 100) -> None:
        """
        监控模型（梯度、参数等）

        Args:
            model: PyTorch模型
            log_freq: 记录频率
        """
        # 默认实现：不做任何事
        pass

    @abstractmethod
    def finish(self) -> None:
        """结束日志记录"""
        pass

    def __enter__(self):
        """Context manager支持"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager支持"""
        self.finish()
