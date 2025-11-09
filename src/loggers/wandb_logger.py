"""
WandB (Weights & Biases) logger实现
"""

from typing import Dict, Any, Optional
from pathlib import Path
import warnings

from .base_logger import BaseLogger


class WandBLogger(BaseLogger):
    """
    WandB日志记录器

    使用示例:
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
        mode: str = "online"
    ):
        """
        Args:
            config: 完整配置字典（会被记录为超参数）
            project: WandB项目名
            entity: WandB用户名/团队名
            name: 运行名称
            tags: 标签列表
            group: 实验组名
            job_type: 任务类型
            mode: 'online', 'offline', or 'disabled'
        """
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            raise ImportError(
                "wandb is not installed. Install it with: pip install wandb"
            )

        # 从配置中提取实验信息
        exp_config = config.get('experiment', {})
        name = name or exp_config.get('name', 'unnamed-run')
        tags = tags or exp_config.get('tags', [])

        # 初始化wandb run
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

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = ""
    ) -> None:
        """
        记录指标到WandB

        Args:
            metrics: {'loss': 0.5, 'auroc': 0.85}
            step: 步数
            prefix: 前缀，如 'train/', 'val/'
        """
        # 添加前缀
        if prefix:
            if not prefix.endswith('/'):
                prefix = prefix + '/'
            formatted_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        else:
            formatted_metrics = metrics

        # 记录到wandb
        self.run.log(formatted_metrics, step=step)

    def log_hyperparameters(self, config: Dict[str, Any]) -> None:
        """
        更新超参数

        Args:
            config: 配置字典
        """
        self.run.config.update(config, allow_val_change=True)

    def log_text(self, key: str, text: str, step: Optional[int] = None) -> None:
        """
        记录文本

        Args:
            key: 文本键名
            text: 文本内容
            step: 步数
        """
        self.run.log({key: self.wandb.Html(text)}, step=step)

    def log_artifact(self, file_path: str, artifact_type: str = "file") -> None:
        """
        上传文件到WandB

        Args:
            file_path: 文件路径
            artifact_type: 文件类型
        """
        artifact = self.wandb.Artifact(
            name=Path(file_path).stem,
            type=artifact_type
        )
        artifact.add_file(file_path)
        self.run.log_artifact(artifact)

    def watch_model(self, model, log_freq: int = 100) -> None:
        """
        监控模型梯度和参数

        Args:
            model: PyTorch模型
            log_freq: 记录频率
        """
        self.run.watch(model, log="all", log_freq=log_freq)

    def finish(self) -> None:
        """结束WandB run"""
        if self.run is not None:
            self.run.finish()
            print("WandB run finished")
