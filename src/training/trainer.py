"""
Step-based CT-CLIP Trainer

Key features:
- Step-based training (not epoch-based)
- Warmup + Cosine LR scheduling
- Multi-GPU support via Accelerator
- Time estimation and progress tracking
- Partial validation for speed
"""

import sys
from pathlib import Path
from datetime import timedelta
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs

from transformers import BertTokenizer

from ..training import get_optimizer, get_warmup_cosine_schedule
from ..data import CTReportDataset
from ..utils import ETACalculator
from ..validation import DiseaseEvaluator
from ..checkpoint import CheckpointManager
from ..loggers import create_logger


def apply_softmax(array):
    """Apply softmax function to tensor"""
    softmax = torch.nn.Softmax(dim=0)
    return softmax(array)


class CTClipTrainer(nn.Module):
    """
    Step-based CT-CLIP Trainer

    All parameters from config, fail fast if missing
    """

    def __init__(self, model, config: Dict):
        """
        Initialize Trainer

        Args:
            model: CT-CLIP model
            config: Configuration dict (required)
        """
        super().__init__()

        self.model = model
        self.config = config

        # Extract config
        data_cfg = config['data']
        training_cfg = config['training']
        validation_cfg = config['validation']
        checkpoint_cfg = config['checkpoint']

        # Training parameters (step-based)
        self.max_steps = training_cfg['max_steps']
        self.batch_size = data_cfg['batch_size']
        self.lr = training_cfg['learning_rate']
        self.wd = training_cfg['weight_decay']
        self.max_grad_norm = training_cfg['max_grad_norm']
        self.gradient_accumulation_steps = training_cfg['gradient_accumulation_steps']

        # Warmup
        self.warmup_steps = training_cfg.get('warmup_steps', 0)
        self.min_lr_ratio = training_cfg.get('min_lr_ratio', 0.01)

        # Validation & Saving
        self.eval_every_n_steps = validation_cfg['eval_every_n_steps']
        self.eval_samples = validation_cfg.get('eval_samples', None)  # None = all samples
        self.save_every_n_steps = training_cfg['save_every_n_steps']
        self.results_folder = Path(checkpoint_cfg['save_dir'])
        self.results_folder.mkdir(parents=True, exist_ok=True)

        # Initialize Accelerator
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, init_kwargs])

        # Tokenizer
        text_cfg = config['model']['text_encoder']
        self.tokenizer = BertTokenizer.from_pretrained(
            text_cfg['path'],
            do_lower_case=text_cfg['do_lower_case']
        )

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_auroc = 0.0

        # Load pathology classes
        self.pathologies = self._load_pathology_classes(data_cfg['labels_valid'])
        self.print(f"Loaded {len(self.pathologies)} pathology classes")

        # Datasets
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

        # DataLoaders
        self.train_dataloader = DataLoader(
            self.train_dataset,
            num_workers=data_cfg['num_workers'],
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.val_dataloader = DataLoader(
            self.val_dataset,
            num_workers=data_cfg['num_workers'],
            batch_size=1,
            shuffle=False,
        )

        # Device
        self.device = self.accelerator.device
        self.model.to(self.device)

        # Optimizer
        all_parameters = set(model.parameters())
        self.optim = get_optimizer(all_parameters, lr=self.lr, wd=self.wd)

        # Scheduler with warmup
        self.scheduler = get_warmup_cosine_schedule(
            self.optim,
            warmup_steps=self.warmup_steps,
            total_steps=self.max_steps,
            min_lr_ratio=self.min_lr_ratio
        )

        # Prepare with Accelerator
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

        # Calculate steps per epoch
        self.steps_per_epoch = len(self.train_dataloader) // self.gradient_accumulation_steps

        # Initialize components
        self.print("\n" + "="*80)
        self.print("Initializing Training Components")
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
        self.print(f"✓ CheckpointManager initialized")

        # ETA Calculator
        self.eta_calculator = ETACalculator(window_size=100)
        self.print("✓ ETA Calculator initialized")

        self.print("="*80 + "\n")

        self.print(f"Trainer initialized:")
        self.print(f"  Train samples: {len(self.train_dataset)}")
        self.print(f"  Val samples: {len(self.val_dataset)}")
        self.print(f"  Max steps: {self.max_steps}")
        self.print(f"  Steps per epoch: {self.steps_per_epoch}")
        self.print(f"  Warmup steps: {self.warmup_steps}")
        if self.eval_samples:
            self.print(f"  Eval samples: {self.eval_samples} (partial validation)")

        # Log hyperparameters
        self.logger.log_hyperparameters(config)

        # Monitor model (WandB)
        if config['logging'].get('use_wandb', False):
            log_freq = config['logging'].get('log_every_n_steps', 100)
            self.logger.watch_model(self.model, log_freq=log_freq)

    def _load_pathology_classes(self, labels_path: str) -> List[str]:
        """Load pathology class names from CSV"""
        df = pd.read_csv(labels_path)
        exclude_cols = {'study_id', 'VolumeName'}
        pathology_cols = [col for col in df.columns if col not in exclude_cols]
        return pathology_cols

    def save_checkpoint(self, metrics: Dict):
        """Save checkpoint"""
        if not self.accelerator.is_local_main_process:
            return

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        save_path = self.checkpoint_manager.save_checkpoint(
            epoch=self.current_epoch,
            global_step=self.global_step,
            model=unwrapped_model,
            optimizer=self.optim,
            scheduler=self.scheduler,
            metrics=metrics,
            config=self.config
        )

        self.print(f'✓ Checkpoint saved: {save_path}')

        # Check if best
        current_auroc = metrics.get('macro_auroc', 0.0)
        if current_auroc > self.best_auroc:
            self.best_auroc = current_auroc
            self.print(f'✅ New best model! AUROC: {self.best_auroc:.4f}')

    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        path = Path(path)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        checkpoint = self.checkpoint_manager.load_checkpoint(
            str(path),
            model=unwrapped_model,
            optimizer=self.optim,
            scheduler=self.scheduler
        )

        self.global_step = checkpoint.get('global_step', 0)
        self.current_epoch = checkpoint['epoch']
        self.best_auroc = checkpoint.get('metrics', {}).get('macro_auroc', 0.0)

        self.print(f"✅ Resumed from step {self.global_step}, epoch {self.current_epoch}")

    def print(self, msg):
        """Print on main process only"""
        self.accelerator.print(msg)

    @property
    def is_main(self):
        """Check if main process"""
        return self.accelerator.is_main_process

    def train_step(self, batch, batch_idx: int) -> float:
        """
        Execute one training step

        Args:
            batch: Data batch
            batch_idx: Batch index

        Returns:
            Loss value
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

        # Update weights after accumulation
        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
            if self.max_grad_norm:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optim.step()
            self.scheduler.step()
            self.optim.zero_grad()

            self.global_step += 1
            self.eta_calculator.update()

        return loss.item() * self.gradient_accumulation_steps

    def validate(self) -> Dict:
        """
        Run validation

        Returns:
            Validation metrics dict
        """
        if not self.is_main:
            return {}

        self.model.eval()

        all_predictions = []
        all_labels = []

        self.print(f"\n{'='*80}")
        self.print(f"Running Validation (Step {self.global_step}, Epoch {self.current_epoch:.2f})")
        self.print(f"{'='*80}")

        # Determine number of samples to evaluate
        num_samples = self.eval_samples if self.eval_samples else len(self.val_dataloader)
        num_samples = min(num_samples, len(self.val_dataloader))

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                if batch_idx >= num_samples:
                    break

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

        # Evaluate
        results = self.evaluator.evaluate(all_predictions, all_labels)

        # Print results
        self.print(f"\n{'='*80}")
        self.print(f"Validation Results (Step {self.global_step}, {num_samples} samples)")
        self.print(f"{'='*80}")
        self.print(f"AUROC: {results['macro_auroc']:.4f}")
        if 'macro_auprc' in results:
            self.print(f"AUPRC: {results['macro_auprc']:.4f}")
        if 'macro_f1' in results:
            self.print(f"F1: {results['macro_f1']:.4f}")
        if 'macro_precision' in results:
            self.print(f"Precision: {results['macro_precision']:.4f}")
        if 'macro_recall' in results:
            self.print(f"Recall: {results['macro_recall']:.4f}")
        self.print(f"{'='*80}\n")

        # Log metrics
        self.logger.log_metrics(results, step=self.global_step, prefix='val')

        return results

    def train(self):
        """Main training loop (step-based)"""
        self.print(f"\n{'='*80}")
        self.print(f"Starting Training")
        self.print(f"{'='*80}\n")

        epoch = 0
        dataloader_iter = iter(self.train_dataloader)
        batch_idx_in_epoch = 0

        while self.global_step < self.max_steps:
            self.model.train()

            # Get next batch
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                # New epoch
                epoch += 1
                self.current_epoch = epoch
                dataloader_iter = iter(self.train_dataloader)
                batch = next(dataloader_iter)
                batch_idx_in_epoch = 0
                self.print(f"\n--- Epoch {epoch} ---")

            # Training step
            loss = self.train_step(batch, batch_idx_in_epoch)
            batch_idx_in_epoch += 1

            # Log every N steps (after accumulation)
            if (batch_idx_in_epoch + 1) % self.gradient_accumulation_steps == 0:
                current_lr = self.optim.param_groups[0]['lr']
                current_epoch_float = self.global_step / self.steps_per_epoch
                avg_step_time = self.eta_calculator.get_avg_step_time()
                eta = self.eta_calculator.get_eta(self.global_step, self.max_steps)
                elapsed = self.eta_calculator.get_elapsed_time()

                if self.global_step % 10 == 0:
                    self.print(
                        f"Step {self.global_step}/{self.max_steps} (Epoch {current_epoch_float:.2f}) | "
                        f"Loss: {loss:.4f} | LR: {current_lr:.2e} | "
                        f"Time/Step: {avg_step_time:.2f}s | ETA: {eta} | Elapsed: {elapsed}"
                    )

                    # Log to logger
                    self.logger.log_metrics({
                        'loss': loss,
                        'learning_rate': current_lr,
                        'epoch': current_epoch_float,
                        'step_time': avg_step_time,
                    }, step=self.global_step, prefix='train')

                # Validation
                if self.global_step > 0 and self.global_step % self.eval_every_n_steps == 0:
                    metrics = self.validate()
                    self.model.train()

                    # Save checkpoint
                    if metrics:
                        self.save_checkpoint(metrics)

                # Save checkpoint (without validation)
                elif self.global_step > 0 and self.global_step % self.save_every_n_steps == 0:
                    metrics = {'macro_auroc': self.best_auroc}
                    self.save_checkpoint(metrics)

        # Final validation
        self.print(f"\n{'='*80}")
        self.print(f"Training Complete!")
        self.print(f"{'='*80}")
        metrics = self.validate()
        if metrics:
            self.save_checkpoint(metrics)

        self.print(f"\nBest AUROC: {self.best_auroc:.4f}")
        self.print(f"Results saved to: {self.results_folder}\n")

        # Finish logging
        self.logger.finish()
