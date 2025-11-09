"""
Checkpoint管理器

功能：
1. 保存完整的训练状态（模型、优化器、调度器、随机数种子等）
2. 自动管理checkpoint数量（只保留最新N个）
3. 保存best checkpoint
4. 支持完全恢复训练
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
    完整的Checkpoint管理器

    用法示例:
        manager = CheckpointManager(
            save_dir="./saves",
            keep_last_n=3,
            save_best=True,
            best_metric="auroc",
            best_metric_mode="max"
        )

        # 保存checkpoint
        manager.save_checkpoint(
            epoch=10,
            global_step=5000,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics={'auroc': 0.85, 'loss': 0.3},
            config=config
        )

        # 恢复checkpoint
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
            save_dir: checkpoint保存目录
            keep_last_n: 保留最新的N个checkpoint (0=保留所有)
            save_best: 是否保存best checkpoint
            best_metric: 用于判断best的指标名
            best_metric_mode: 'max' or 'min'
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.best_metric = best_metric
        self.best_metric_mode = best_metric_mode

        # 初始化best值
        if best_metric_mode == 'max':
            self.best_value = -float('inf')
        elif best_metric_mode == 'min':
            self.best_value = float('inf')
        else:
            raise ValueError(f"best_metric_mode must be 'max' or 'min', got {best_metric_mode}")

        # 记录checkpoint历史
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
        保存完整checkpoint

        Args:
            epoch: 当前epoch
            global_step: 全局步数
            model: PyTorch模型
            optimizer: 优化器
            scheduler: 学习率调度器
            metrics: 验证指标
            config: 配置字典
            extra_state: 额外需要保存的状态

        Returns:
            保存的文件路径
        """
        metrics = metrics or {}
        config = config or {}
        extra_state = extra_state or {}

        # 构建checkpoint字典
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': config,
            'extra_state': extra_state
        }

        # 保存调度器状态
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # 保存随机数状态（用于完全可复现）
        checkpoint['random_states'] = self._get_random_states()

        # 保存latest checkpoint
        latest_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, latest_path)

        print(f"Saved checkpoint: {latest_path}")

        # 更新历史记录
        self.checkpoint_history.append({
            'path': latest_path,
            'epoch': epoch,
            'metrics': metrics
        })

        # 管理checkpoint数量
        if self.keep_last_n > 0:
            self._cleanup_old_checkpoints()

        # 保存best checkpoint
        if self.save_best:
            self._save_best_if_needed(checkpoint, metrics, epoch)

        # 始终保留一个latest链接
        self._update_latest_symlink(latest_path)

        return str(latest_path)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        load_random_states: bool = True,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        加载checkpoint并恢复训练状态

        Args:
            checkpoint_path: checkpoint文件路径 (或 'latest', 'best')
            model: PyTorch模型
            optimizer: 优化器 (None=不恢复优化器状态)
            scheduler: 调度器 (None=不恢复调度器状态)
            load_random_states: 是否恢复随机数状态
            strict: 是否严格匹配model state dict

        Returns:
            checkpoint字典，包含epoch, global_step, metrics等
        """
        # 处理特殊路径
        if checkpoint_path == 'latest':
            checkpoint_path = self.save_dir / "latest.pt"
        elif checkpoint_path == 'best':
            checkpoint_path = self.save_dir / "best.pt"
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint from: {checkpoint_path}")

        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 恢复模型
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        print(f"✓ Model state restored")

        # 恢复优化器
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"✓ Optimizer state restored")

        # 恢复调度器
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"✓ Scheduler state restored")

        # 恢复随机数状态
        if load_random_states and 'random_states' in checkpoint:
            self._set_random_states(checkpoint['random_states'])
            print(f"✓ Random states restored")

        # 打印恢复信息
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
        如果指标更好，保存为best checkpoint

        Args:
            checkpoint: checkpoint字典
            metrics: 验证指标
            epoch: 当前epoch
        """
        # 提取目标指标
        # 支持 'auroc' 或 'macro_auroc'
        metric_value = None
        for key in [self.best_metric, f'macro_{self.best_metric}']:
            if key in metrics:
                metric_value = metrics[key]
                break

        if metric_value is None:
            warnings.warn(f"Metric '{self.best_metric}' not found in metrics, skipping best checkpoint")
            return

        # 判断是否更好
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
        """删除旧的checkpoints，只保留最新的N个"""
        if len(self.checkpoint_history) <= self.keep_last_n:
            return

        # 按epoch排序
        sorted_history = sorted(self.checkpoint_history, key=lambda x: x['epoch'])

        # 删除最旧的
        to_delete = sorted_history[:-self.keep_last_n]

        for item in to_delete:
            path = item['path']
            if path.exists() and path.name != 'best.pt' and path.name != 'latest.pt':
                path.unlink()
                print(f"Deleted old checkpoint: {path}")

        # 更新历史记录
        self.checkpoint_history = sorted_history[-self.keep_last_n:]

    def _update_latest_symlink(self, latest_path: Path) -> None:
        """创建或更新latest符号链接"""
        symlink_path = self.save_dir / "latest.pt"

        # 删除旧的符号链接
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()

        # 创建新的符号链接（如果系统支持）
        try:
            symlink_path.symlink_to(latest_path.name)
        except OSError:
            # 不支持符号链接（如Windows），直接复制
            import shutil
            shutil.copy(latest_path, symlink_path)

    def _get_random_states(self) -> Dict[str, Any]:
        """获取所有随机数生成器的状态"""
        states = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
        }

        # CUDA随机数状态
        if torch.cuda.is_available():
            states['cuda'] = torch.cuda.get_rng_state_all()

        return states

    def _set_random_states(self, states: Dict[str, Any]) -> None:
        """恢复所有随机数生成器的状态"""
        random.setstate(states['python'])
        np.random.set_state(states['numpy'])
        torch.set_rng_state(states['torch'])

        if torch.cuda.is_available() and 'cuda' in states:
            torch.cuda.set_rng_state_all(states['cuda'])

    def list_checkpoints(self) -> list:
        """列出所有可用的checkpoints"""
        checkpoints = sorted(self.save_dir.glob("checkpoint_epoch_*.pt"))
        return [str(p) for p in checkpoints]

    def get_latest_checkpoint(self) -> Optional[str]:
        """获取最新的checkpoint路径"""
        latest_path = self.save_dir / "latest.pt"
        if latest_path.exists():
            return str(latest_path)
        return None

    def get_best_checkpoint(self) -> Optional[str]:
        """获取best checkpoint路径"""
        best_path = self.save_dir / "best.pt"
        if best_path.exists():
            return str(best_path)
        return None
