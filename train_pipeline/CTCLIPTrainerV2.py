"""
CT-CLIP Trainer V2 - 简洁版

核心原则:
- 配置驱动: 所有参数从 config 读取
- 快速失败: 缺少配置直接报错，不使用默认值
- 极简设计: 只使用 V2 系统，无向后兼容代码

用法:
    config = load_config('configs/base_config.yaml')
    model = build_model(config)
    trainer = CTClipTrainerV2(model, config)
    trainer.train()
"""

import sys
import os
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pathlib import Path
from datetime import timedelta
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np
import pandas as pd

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs

from transformers import BertTokenizer

from transformer_maskgit.optimizer import get_optimizer
from src.load_ctreport_dataset import CTReportDataset
from ct_clip import CTCLIP

# V2 components
from loggers import create_logger
from validation import DiseaseEvaluator
from checkpoint import CheckpointManager


def apply_softmax(array):
    """Applies softmax function to a torch array."""
    softmax = torch.nn.Softmax(dim=0)
    return softmax(array)


class CTClipTrainerV2(nn.Module):
    """
    CT-CLIP Trainer V2 - 配置驱动的简洁版本

    特点:
    - 所有参数从 config 读取
    - 使用 V2 系统: logger, evaluator, checkpoint_manager
    - 快速失败: 缺少必需配置会立即报错
    - 无默认值: 强制显式配置
    """

    def __init__(self, model: CTCLIP, config: Dict):
        """
        初始化 Trainer V2

        Args:
            model: CT-CLIP 模型
            config: 配置字典（必需）
                必需包含: experiment, data, model, training, validation, checkpoint, logging

        如果缺少任何必需字段，会直接报错（快速失败）
        """
        super().__init__()

        self.model = model
        self.config = config

        # ===== 从 config 读取参数（缺少就报错）=====
        data_cfg = config['data']
        training_cfg = config['training']
        validation_cfg = config['validation']
        checkpoint_cfg = config['checkpoint']

        # Training parameters
        self.num_epochs = training_cfg['num_epochs']
        self.batch_size = data_cfg['batch_size']
        self.lr = training_cfg['learning_rate']
        self.wd = training_cfg['weight_decay']
        self.max_grad_norm = training_cfg['max_grad_norm']
        self.gradient_accumulation_steps = training_cfg['gradient_accumulation_steps']

        # Validation & Saving
        self.validate_every_n_epochs = validation_cfg['every_n_epochs']
        self.save_every_n_epochs = training_cfg['save_every_n_epochs']
        self.results_folder = Path(checkpoint_cfg['save_dir'])
        self.results_folder.mkdir(parents=True, exist_ok=True)

        # ===== Initialize Accelerator =====
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, init_kwargs])

        # ===== Tokenizer =====
        text_cfg = config['model']['text_encoder']
        self.tokenizer = BertTokenizer.from_pretrained(
            text_cfg['path'],
            do_lower_case=text_cfg['do_lower_case']
        )

        # ===== Training State =====
        self.current_epoch = 0
        self.global_step = 0
        self.best_auroc = 0.0

        # ===== Load Pathology Classes =====
        self.pathologies = self._load_pathology_classes(data_cfg['labels_valid'])
        self.print(f"Loaded {len(self.pathologies)} pathology classes: {self.pathologies}")

        # ===== Datasets =====
        self.train_dataset = CTReportDataset(
            data_folder=data_cfg['train_dir'],
            reports_file=data_cfg['reports_train'],
            meta_file=data_cfg['train_meta'],
            labels=data_cfg['labels_train'],
            mode="train"
        )

        self.val_dataset = CTReportDataset(
            data_folder=data_cfg['valid_dir'],
            reports_file=data_cfg['reports_valid'],
            meta_file=data_cfg['valid_meta'],
            labels=data_cfg['labels_valid'],
            mode="val"
        )

        # ===== DataLoaders =====
        self.train_dataloader = DataLoader(
            self.train_dataset,
            num_workers=data_cfg['num_workers'],
            batch_size=self.batch_size,
            # shuffle=True,
            shuffle=False,
        )

        self.val_dataloader = DataLoader(
            self.val_dataset,
            num_workers=data_cfg['num_workers'],
            batch_size=1,  # Validation batch size = 1
            shuffle=False,
        )

        # ===== Device =====
        self.device = self.accelerator.device
        self.model.to(self.device)

        # ===== Optimizer =====
        all_parameters = set(model.parameters())
        self.optim = get_optimizer(all_parameters, lr=self.lr, wd=self.wd)

        # ===== Learning Rate Scheduler =====
        total_steps = len(self.train_dataloader) * self.num_epochs // self.gradient_accumulation_steps
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optim,
            T_max=total_steps,
            eta_min=self.lr * 0.01
        )

        # ===== Prepare with Accelerator =====
        (
            self.train_dataloader,
            self.val_dataloader,
            self.model,
            self.optim,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.train_dataloader,
            self.val_dataloader,
            self.model,
            self.optim,
            self.scheduler,
        )

        # ===== V2 Components (直接初始化，不判断) =====
        self.print("\n" + "="*80)
        self.print("Initializing V2 Systems")
        self.print("="*80)

        # Logger
        self.logger = create_logger(config)
        self.print("✓ Logger initialized")

        # Evaluator
        self.evaluator = DiseaseEvaluator(
            pathology_classes=self.pathologies,
            metrics=validation_cfg['metrics'],
            use_bootstrap=validation_cfg['use_bootstrap'],
            threshold=validation_cfg['threshold']
        )
        self.print(f"✓ Evaluator initialized with metrics: {validation_cfg['metrics']}")

        # Checkpoint Manager
        self.checkpoint_manager = CheckpointManager(
            save_dir=checkpoint_cfg['save_dir'],
            keep_last_n=checkpoint_cfg['keep_last_n'],
            save_best=checkpoint_cfg['save_best'],
            best_metric=checkpoint_cfg['best_metric'],
            best_metric_mode=checkpoint_cfg['best_metric_mode']
        )
        self.print(f"✓ CheckpointManager initialized (save_dir: {checkpoint_cfg['save_dir']})")
        self.print("="*80 + "\n")

        self.print(f"Trainer initialized: {len(self.train_dataset)} train samples, "
                   f"{len(self.val_dataset)} val samples")
        self.print(f"Training for {self.num_epochs} epochs, {total_steps} total steps")

        # ===== 监控模型和记录超参数 =====
        # 记录超参数到 logger (WandB/TensorBoard)
        self.logger.log_hyperparameters(config)
        self.print("✓ Hyperparameters logged")

        # 监控模型梯度和参数 (WandB)
        if config['logging'].get('use_wandb', False):
            log_freq = config['logging'].get('log_every_n_steps', 100)
            self.logger.watch_model(self.model, log_freq=log_freq)
            self.print(f"✓ Model monitoring enabled (log_freq={log_freq})")

    def _load_pathology_classes(self, labels_path: str) -> List[str]:
        """
        从 CSV 文件加载病理类别名称

        Args:
            labels_path: 标签文件路径

        Returns:
            病理类别列表

        注意: 如果文件不存在或读取失败，会直接报错（快速失败）
        """
        # 直接读取，失败就让它报错
        df = pd.read_csv(labels_path)

        # 排除元数据列
        exclude_cols = {
            'study_id', 'VolumeName'  }

        pathology_cols = [col for col in df.columns if col not in exclude_cols]

        return pathology_cols

    def save_checkpoint(self, epoch: int, metrics: Dict):
        """
        保存 checkpoint（只使用 V2 CheckpointManager）

        Args:
            epoch: 当前 epoch
            metrics: 验证指标
        """
        if not self.accelerator.is_local_main_process:
            return

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        save_path = self.checkpoint_manager.save_checkpoint(
            epoch=epoch,
            global_step=self.global_step,
            model=unwrapped_model,
            optimizer=self.optim,
            scheduler=self.scheduler,
            metrics=metrics,
            config=self.config
        )

        self.print(f'✓ Checkpoint saved: {save_path}')

        # 判断是否是 best
        current_auroc = metrics.get('macro_auroc', 0.0)
        if current_auroc > self.best_auroc:
            self.best_auroc = current_auroc
            self.print(f'✅ New best model! AUROC: {self.best_auroc:.4f}')

    def load_checkpoint(self, path: str):
        """
        加载 checkpoint（只使用 V2 CheckpointManager）

        Args:
            path: checkpoint 路径
        """
        path = Path(path)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        checkpoint = self.checkpoint_manager.load_checkpoint(
            str(path),
            model=unwrapped_model,
            optimizer=self.optim,
            scheduler=self.scheduler
        )

        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_auroc = checkpoint.get('metrics', {}).get('macro_auroc', 0.0)

        self.print(f"✅ Resumed from epoch {checkpoint['epoch']}, step {self.global_step}")

    def print(self, msg):
        """Print message (only on main process)."""
        self.accelerator.print(msg)

    @property
    def is_main(self):
        """Check if current process is main."""
        return self.accelerator.is_main_process

    def train_step(self, batch, batch_idx: int) -> float:
        """
        执行一个训练步骤

        Args:
            batch: 数据批次
            batch_idx: 批次索引

        Returns:
            loss 值
        """
        device = self.device

        # Unpack batch
        volume_tensor, report_text, disease_labels, study_id, embed_tensor = batch
        volume_tensor = volume_tensor.to(device)

        # Tokenize text
        report_text = list(report_text)
        text_tokens = self.tokenizer(
            report_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        ).to(device)

        # Forward pass
        with self.accelerator.autocast():
            loss = self.model(text_tokens, volume_tensor, return_loss=True, device=device)
            loss = loss / self.gradient_accumulation_steps

        # Backward pass
        self.accelerator.backward(loss)

        # Update weights (only after accumulation steps)
        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
            if self.max_grad_norm:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optim.step()
            self.scheduler.step()
            self.optim.zero_grad()

            self.global_step += 1

        return loss.item() * self.gradient_accumulation_steps

    def validate(self) -> Dict:
        """
        运行验证（只使用 V2 DiseaseEvaluator）

        Returns:
            验证指标字典
        """
        if not self.is_main:
            return {}

        self.model.eval()

        all_predictions = []
        all_labels = []

        self.print(f"\n{'='*80}")
        self.print(f"Running Validation (Epoch {self.current_epoch})")
        self.print(f"{'='*80}")

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                volume_tensor, report_text, disease_labels, study_id, embed_tensor = batch
                volume_tensor = volume_tensor.to(self.device)

                # Predict for each pathology
                predicted_labels = []
                for pathology in self.pathologies:
                    texts = [f"There is {pathology}.", f"There is no {pathology}."]
                    text_tokens = self.tokenizer(
                        texts,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=512
                    ).to(self.device)

                    output = self.model(text_tokens, volume_tensor, device=self.device)
                    output = apply_softmax(output)

                    predicted_labels.append(output[0].detach().cpu().numpy())

                all_predictions.append(predicted_labels)
                all_labels.append(disease_labels.detach().cpu().numpy()[0])

        # Convert to numpy arrays
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)

        # Use V2 Evaluator
        results = self.evaluator.evaluate(all_predictions, all_labels)

        # Extract metrics
        mean_auroc = results['macro_auroc']
        mean_auprc = results.get('macro_auprc', 0.0)
        mean_f1 = results.get('macro_f1', 0.0)

        # Print results
        self.print(f"\n{'='*80}")
        self.print(f"Validation Results (Epoch {self.current_epoch})")
        self.print(f"{'='*80}")
        self.print(f"AUROC: {mean_auroc:.4f}")
        if 'macro_auprc' in results:
            self.print(f"AUPRC: {mean_auprc:.4f}")
        if 'macro_f1' in results:
            self.print(f"F1: {mean_f1:.4f}")
        self.print(f"{'='*80}\n")

        # Log to V2 logger
        self.logger.log_metrics(results, step=self.global_step, prefix='val')

        return results

    def train(self):
        """
        主训练循环
        """
        self.print(f"\n{'='*80}")
        self.print(f"Starting Training")
        self.print(f"{'='*80}\n")

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            self.model.train()

            epoch_loss = 0.0
            num_batches = 0

            self.print(f"\n--- Epoch {epoch + 1}/{self.num_epochs} ---")

            # Training loop
            for batch_idx, batch in enumerate(self.train_dataloader):
                loss = self.train_step(batch, batch_idx)
                epoch_loss += loss
                num_batches += 1

                # Log every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    avg_loss = epoch_loss / num_batches
                    current_lr = self.optim.param_groups[0]['lr']
                    self.print(
                        f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(self.train_dataloader)} | "
                        f"Loss: {loss:.4f} | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e}"
                    )

                    # Log to V2 logger
                    self.logger.log_metrics({
                        'loss': loss,
                        'avg_loss': avg_loss,
                        'learning_rate': current_lr
                    }, step=self.global_step, prefix='train')



            # Epoch summary
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            self.print(f"\nEpoch {epoch + 1} completed | Avg Loss: {avg_epoch_loss:.4f}")

            # Validation
            metrics = None
            if (epoch + 1) % self.validate_every_n_epochs == 0:
                metrics = self.validate()

            # Save checkpoint
            if (epoch + 1) % self.save_every_n_epochs == 0:
                if metrics is None:
                    metrics = {'macro_auroc': self.best_auroc}
                self.save_checkpoint(epoch, metrics)

        self.print(f"\n{'='*80}")
        self.print(f"Training Complete!")
        self.print(f"Best AUROC: {self.best_auroc:.4f}")
        self.print(f"{'='*80}\n")

        # Finish logging
        self.logger.finish()
