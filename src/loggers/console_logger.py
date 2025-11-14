"""
控制台日志记录器

简单的打印到stdout
"""

from typing import Dict, Any, Optional
from datetime import datetime

from .base_logger import BaseLogger


class ConsoleLogger(BaseLogger):
    """
    控制台日志记录器

    简单地打印指标到标准输出
    """

    def __init__(self, print_timestamp: bool = True):
        """
        Args:
            print_timestamp: 是否打印时间戳
        """
        self.print_timestamp = print_timestamp

    def _format_timestamp(self) -> str:
        """生成时间戳前缀"""
        if self.print_timestamp:
            return f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        return ""

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = ""
    ) -> None:
        """
        打印指标

        Args:
            metrics: 指标字典
            step: 步数
            prefix: 前缀
        """
        timestamp = self._format_timestamp()

        # 构建步数信息
        step_info = f"Step {step}" if step is not None else ""

        # 构建前缀
        prefix_info = f"{prefix}" if prefix else ""

        # 组合header
        header_parts = [p for p in [step_info, prefix_info] if p]
        header = " - ".join(header_parts)

        # 格式化指标
        metrics_parts = []
        for k, v in metrics.items():
            if isinstance(v, dict):
                # Skip nested dicts in console output (too verbose)
                continue
            elif isinstance(v, (int, float)):
                # Format numbers with 4 decimal places
                metrics_parts.append(f"{k}: {v:.4f}")
            else:
                # Other types: convert to string
                metrics_parts.append(f"{k}: {v}")

        metrics_str = ", ".join(metrics_parts)

        # 打印
        if header:
            print(f"{timestamp}{header} | {metrics_str}")
        else:
            print(f"{timestamp}{metrics_str}")

    def log_hyperparameters(self, config: Dict[str, Any]) -> None:
        """
        打印配置

        Args:
            config: 配置字典
        """
        timestamp = self._format_timestamp()
        print(f"{timestamp}Hyperparameters:")
        self._print_dict(config, indent=2)

    def _print_dict(self, d: Dict, indent: int = 0):
        """递归打印字典"""
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                self._print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")

    def log_text(self, key: str, text: str, step: Optional[int] = None) -> None:
        """
        打印文本

        Args:
            key: 文本键
            text: 文本内容
            step: 步数
        """
        timestamp = self._format_timestamp()
        step_info = f"Step {step} - " if step is not None else ""
        print(f"{timestamp}{step_info}{key}:")
        print(text)

    def finish(self) -> None:
        """结束日志（console logger不需要做anything）"""
        pass
