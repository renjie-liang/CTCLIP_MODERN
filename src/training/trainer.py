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
from datetime import datetime, timedelta
from typing import Dict, List
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import pandas as pd

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs

from transformers import AutoTokenizer

from ..training import get_optimizer, get_warmup_cosine_schedule
from ..data.webdataset_loader import CTReportWebDataset
from ..utils import ETACalculator, get_memory_info, get_detailed_gpu_memory_info
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

        # Basic training parameters
        self.batch_size = data_cfg['batch_size']
        self.lr = training_cfg['learning_rate']
        self.wd = training_cfg['weight_decay']
        self.max_grad_norm = training_cfg['max_grad_norm']
        self.gradient_accumulation_steps = training_cfg['gradient_accumulation_steps']
        self.min_lr_ratio = training_cfg.get('min_lr_ratio', 0.01)

        # Store configs for later use (after dataset is loaded)
        self._training_cfg = training_cfg
        self._validation_cfg = validation_cfg

        self.results_folder = Path(checkpoint_cfg['save_dir'])
        self.results_folder.mkdir(parents=True, exist_ok=True)

        # Initialize Accelerator with mixed precision training
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs, init_kwargs],
            mixed_precision='fp16'  # Enable automatic mixed precision (AMP)
        )

        # Tokenizer
        text_cfg = config['model']['text_encoder']
        self.tokenizer = AutoTokenizer.from_pretrained(
            text_cfg['path'],
            do_lower_case=text_cfg['do_lower_case'],
            trust_remote_code=True
        )

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_auroc = 0.0

        # Performance profiling
        self.profile_timing = config['logging'].get('profile_timing', False)
        if self.profile_timing:
            self.print("⚠ Performance profiling enabled - detailed timing will be printed")
            # Initialize timing accumulators
            self.timing_stats = {
                'data_loading': [],
                'data_to_device': [],
                'tokenize': [],
                'forward': [],
                'backward': [],
                'optimizer_step': [],
                'total_step': []
            }
            # Enable model internal profiling
            if hasattr(self.model, 'visual_transformer'):
                self.model.visual_transformer.profile_timing = True

        # Load pathology classes
        self.pathologies = self._load_pathology_classes(data_cfg['labels_valid'])
        self.print(f"Loaded {len(self.pathologies)} pathology classes")

        # WebDataset format (using float16 for optimized I/O)
        self.print(f"Using WebDataset format")

        # Training dataset
        self.train_dataset = CTReportWebDataset(
            shard_pattern=data_cfg['webdataset_shards_train'],
            shuffle=True,
            buffer_size=data_cfg.get('shuffle_buffer_size', 1000),
            mode="train"
        )

        # Validation dataset
        self.val_dataset = CTReportWebDataset(
            shard_pattern=data_cfg['webdataset_shards_val'],
            shuffle=False,
            buffer_size=0,
            mode="val"
        )

        # Create DataLoaders
        self.train_dataloader = self.train_dataset.create_pytorch_dataloader(
            batch_size=self.batch_size,
            num_workers=data_cfg['num_workers'],
            prefetch_factor=data_cfg.get('prefetch_factor', 2)
        )

        self.val_dataloader = self.val_dataset.create_pytorch_dataloader(
            batch_size=1,
            num_workers=data_cfg['num_workers'],
            prefetch_factor=data_cfg.get('prefetch_factor', 2)
        )

        # Calculate steps per epoch FIRST (needed for epoch-based configs)
        # WebDataset doesn't have len(), so calculate from num_samples
        if self.train_dataset.num_samples is not None:
            self.steps_per_epoch = (self.train_dataset.num_samples // self.batch_size) // self.gradient_accumulation_steps
        else:
            # Fallback: estimate based on typical dataset size
            self.steps_per_epoch = 1000
            self.print("Warning: Could not determine dataset size, using estimated steps_per_epoch=1000")

        # Calculate training duration (prioritize max_steps over max_epochs)
        training_cfg = self._training_cfg
        validation_cfg = self._validation_cfg

        if training_cfg.get('max_steps') is not None:
            # Step-based training
            self.max_steps = training_cfg['max_steps']
            self.max_epochs_config = self.max_steps / self.steps_per_epoch
            self.print(f"Using step-based training: {self.max_steps} steps (~{self.max_epochs_config:.2f} epochs)")
        else:
            # Epoch-based training
            self.max_epochs_config = training_cfg['max_epochs']
            self.max_steps = int(self.max_epochs_config * self.steps_per_epoch)
            self.print(f"Using epoch-based training: {self.max_epochs_config} epochs = {self.max_steps} steps")
            self.print(f"  Dataset: {self.train_dataset.num_samples} samples / batch_size {self.batch_size} = {self.steps_per_epoch} steps/epoch")

        # Calculate warmup steps (prioritize warmup_steps over warmup_epochs)
        if training_cfg.get('warmup_steps') is not None:
            self.warmup_steps = training_cfg['warmup_steps']
        else:
            warmup_epochs = training_cfg.get('warmup_epochs', 0)
            self.warmup_steps = int(warmup_epochs * self.steps_per_epoch)

        # Calculate evaluation frequency (prioritize steps over epochs)
        if validation_cfg.get('eval_every_n_steps') is not None:
            self.eval_every_n_steps = validation_cfg['eval_every_n_steps']
        else:
            eval_every_n_epochs = validation_cfg.get('eval_every_n_epochs', 1.0)
            self.eval_every_n_steps = int(eval_every_n_epochs * self.steps_per_epoch)

        self.eval_samples = validation_cfg.get('eval_samples', None)  # None = all samples

        # Calculate checkpoint saving frequency (prioritize steps over epochs)
        if training_cfg.get('save_every_n_steps') is not None:
            self.save_every_n_steps = training_cfg['save_every_n_steps']
        else:
            save_every_n_epochs = training_cfg.get('save_every_n_epochs', 1.0)
            self.save_every_n_steps = int(save_every_n_epochs * self.steps_per_epoch)

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

        # Prepare model, optimizer, scheduler (skip dataloaders to preserve custom collate_fn)
        self.model, self.optim, self.scheduler = self.accelerator.prepare(
            self.model, self.optim, self.scheduler
        )

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
        self.print(f"  Batch size: {self.batch_size}")
        self.print(f"  Steps per epoch: {self.steps_per_epoch}")
        self.print(f"")
        self.print(f"  Training duration: {self.max_epochs_config:.2f} epochs = {self.max_steps} steps")
        self.print(f"  Warmup: {self.warmup_steps} steps ({self.warmup_steps/self.steps_per_epoch:.2f} epochs)")
        self.print(f"  Eval every: {self.eval_every_n_steps} steps ({self.eval_every_n_steps/self.steps_per_epoch:.2f} epochs)")
        self.print(f"  Save every: {self.save_every_n_steps} steps ({self.save_every_n_steps/self.steps_per_epoch:.2f} epochs)")
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

    def train_step(self, batch, batch_idx: int) -> Dict:
        """
        Execute one training step

        Args:
            batch: Data batch
            batch_idx: Batch index

        Returns:
            Dict with loss and timing info
        """
        timing = {}
        device = self.device

        # Unpack batch and move to device
        if self.profile_timing:
            torch.cuda.synchronize()
            t0 = time.time()

        volume_tensor, report_text, disease_labels, study_id, embed_tensor = batch
        volume_tensor = volume_tensor.to(device, non_blocking=True)

        if self.profile_timing:
            torch.cuda.synchronize()
            timing['data_to_device'] = time.time() - t0

        # Tokenize text
        if self.profile_timing:
            torch.cuda.synchronize()
            t0 = time.time()

        report_text = list(report_text)
        text_tokens = self.tokenizer(
            report_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        ).to(device, non_blocking=True)

        if self.profile_timing:
            torch.cuda.synchronize()
            timing['tokenize'] = time.time() - t0

        # Forward pass with automatic mixed precision (float32 inputs → float16 computation)
        if self.profile_timing:
            torch.cuda.synchronize()
            t0 = time.time()

        with self.accelerator.autocast():
            loss = self.model(text_tokens, volume_tensor, return_loss=True, device=device)
            loss = loss / self.gradient_accumulation_steps

        if self.profile_timing:
            torch.cuda.synchronize()
            timing['forward'] = time.time() - t0

        # Backward pass
        if self.profile_timing:
            torch.cuda.synchronize()
            t0 = time.time()

        self.accelerator.backward(loss)

        if self.profile_timing:
            torch.cuda.synchronize()
            timing['backward'] = time.time() - t0

        # Update weights after accumulation
        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
            if self.profile_timing:
                torch.cuda.synchronize()
                t0 = time.time()

            if self.max_grad_norm:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optim.step()
            self.scheduler.step()
            self.optim.zero_grad()

            if self.profile_timing:
                torch.cuda.synchronize()
                timing['optimizer_step'] = time.time() - t0

            self.global_step += 1
            self.eta_calculator.update()

        return {
            'loss': loss.item() * self.gradient_accumulation_steps,
            'timing': timing
        }

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
        # For WebDataset with batch_size=1, num_batches == num_samples
        total_val_samples = len(self.val_dataset)
        num_samples = self.eval_samples if self.eval_samples else total_val_samples
        num_samples = min(num_samples, total_val_samples)

        with torch.no_grad():
            # Add progress bar
            pbar = tqdm(
                enumerate(self.val_dataloader),
                total=num_samples,
                desc=f"Validating",
                disable=not self.is_main
            )

            for batch_idx, batch in pbar:
                if batch_idx >= num_samples:
                    break

                volume_tensor, report_text, disease_labels, study_id, embed_tensor = batch
                volume_tensor = volume_tensor.to(self.device)

                # Batch all pathology texts together for efficient inference
                all_texts = []
                for pathology in self.pathologies:
                    all_texts.extend([f"There is {pathology}.", f"There is no {pathology}."])

                # Single tokenization and forward pass for all pathologies
                text_tokens = self.tokenizer(
                    all_texts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512
                ).to(self.device)

                output = self.model(text_tokens, volume_tensor, device=self.device)

                # Reshape output: (36,) -> (18, 2)
                output = output.reshape(len(self.pathologies), 2)

                # Apply softmax for each pathology pair (along dim=1)
                softmax = torch.nn.Softmax(dim=1)
                output = softmax(output)

                # Take positive class (index 0: "There is {pathology}")
                predicted_labels = output[:, 0].detach().cpu().numpy().tolist()

                all_predictions.append(predicted_labels)

                # Handle both tensor and numpy array inputs (WebDataset returns numpy)
                if isinstance(disease_labels, torch.Tensor):
                    all_labels.append(disease_labels.detach().cpu().numpy()[0])
                else:
                    # Already numpy array from WebDataset
                    all_labels.append(disease_labels[0])

                # Update progress bar
                pbar.set_postfix({'samples': f'{batch_idx+1}/{num_samples}'})

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

            # Get next batch (measure TOTAL step time including data loading)
            # Always record time (minimal overhead), but only sync GPU if profiling
            if self.profile_timing:
                torch.cuda.synchronize()  # Only sync when profiling (expensive)
            step_start = time.time()  # Start timing BEFORE data loading
            data_load_start = time.time()

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

            # Record data loading time (always, minimal overhead)
            data_load_time = time.time() - data_load_start

            # Training step
            result = self.train_step(batch, batch_idx_in_epoch)
            loss = result['loss']
            step_timing = result['timing']
            batch_idx_in_epoch += 1

            # Calculate total step time (always)
            # Only sync GPU when profiling for precise timing
            if self.profile_timing:
                torch.cuda.synchronize()  # Precise GPU timing (expensive)
            total_step_time = time.time() - step_start  # Total = data loading + GPU ops

            # Log every N steps (after accumulation)
            if (batch_idx_in_epoch + 1) % self.gradient_accumulation_steps == 0:
                current_lr = self.optim.param_groups[0]['lr']
                current_epoch_float = self.global_step / self.steps_per_epoch
                avg_step_time = self.eta_calculator.get_avg_step_time()
                eta = self.eta_calculator.get_eta(self.global_step, self.max_steps)
                elapsed = self.eta_calculator.get_elapsed_time()

                if self.global_step % 10 == 0:
                    # Calculate model time (total - data loading)
                    model_time = total_step_time - data_load_time

                    # Simple one-line log (always shown)
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    self.print(
                        f"[{timestamp}] Ep {current_epoch_float:.2f} | Step {self.global_step}/{self.max_steps} | "
                        f"loss: {loss:.4f} | lr: {current_lr:.2e} | "
                        f"time: {total_step_time:.2f}s (data: {data_load_time:.2f}s, model: {model_time:.2f}s) | "
                        f"ETA: {eta} | elapsed: {elapsed}"
                    )

                    # Collect timing stats for averaging (only when profiling)
                    if self.profile_timing:
                        self.timing_stats['data_loading'].append(data_load_time)
                        self.timing_stats['total_step'].append(total_step_time)
                        for key, value in step_timing.items():
                            if key in self.timing_stats:
                                self.timing_stats[key].append(value)

                        # Detailed breakdown every 100 steps
                        if self.global_step % 100 == 0 and len(self.timing_stats['total_step']) > 0:
                            self.print(f"  📊 Timing Breakdown (avg last 100 steps):")

                            # Calculate averages
                            avg_total = np.mean(self.timing_stats['total_step'][-100:])
                            avg_data_load = np.mean(self.timing_stats['data_loading'][-100:])
                            avg_data_device = np.mean(self.timing_stats['data_to_device'][-100:]) if self.timing_stats['data_to_device'] else 0
                            avg_tokenize = np.mean(self.timing_stats['tokenize'][-100:]) if self.timing_stats['tokenize'] else 0
                            avg_forward = np.mean(self.timing_stats['forward'][-100:]) if self.timing_stats['forward'] else 0
                            avg_backward = np.mean(self.timing_stats['backward'][-100:]) if self.timing_stats['backward'] else 0
                            avg_optim = np.mean(self.timing_stats['optimizer_step'][-100:]) if self.timing_stats['optimizer_step'] else 0

                            # Get CT-VIT internal timing (if available)
                            ctvit_timing = {}
                            try:
                                unwrapped_model = self.accelerator.unwrap_model(self.model)
                                if hasattr(unwrapped_model, 'visual_transformer') and hasattr(unwrapped_model.visual_transformer, 'timing_buffer'):
                                    ctvit_timing = unwrapped_model.visual_transformer.timing_buffer.copy()
                            except:
                                pass

                            # Data Loading breakdown
                            self.print(f"     Data Loading:           {avg_data_load*1000:7.2f}ms ({avg_data_load/avg_total*100:5.1f}%)")
                            if avg_data_device > 0:
                                self.print(f"       ├─ Data to Device:    {avg_data_device*1000:7.2f}ms ({avg_data_device/avg_total*100:5.1f}%)")
                            if avg_tokenize > 0:
                                self.print(f"       └─ Tokenize:          {avg_tokenize*1000:7.2f}ms ({avg_tokenize/avg_total*100:5.1f}%)")

                            self.print("")

                            # Model Forward breakdown
                            self.print(f"     Model Forward:          {avg_forward*1000:7.2f}ms ({avg_forward/avg_total*100:5.1f}%)")
                            if ctvit_timing:
                                spatial_time = ctvit_timing.get('spatial_transformer', 0)
                                temporal_time = ctvit_timing.get('temporal_transformer', 0)
                                other_time = avg_forward - spatial_time - temporal_time
                                if spatial_time > 0:
                                    self.print(f"       ├─ Spatial Trans:     {spatial_time*1000:7.2f}ms ({spatial_time/avg_total*100:5.1f}%)")
                                if temporal_time > 0:
                                    self.print(f"       ├─ Temporal Trans:    {temporal_time*1000:7.2f}ms ({temporal_time/avg_total*100:5.1f}%)")
                                if other_time > 0:
                                    self.print(f"       └─ Other ops:         {other_time*1000:7.2f}ms ({other_time/avg_total*100:5.1f}%)")

                            self.print("")

                            # Backward and Optimizer
                            if avg_backward > 0:
                                self.print(f"     Backward Pass:          {avg_backward*1000:7.2f}ms ({avg_backward/avg_total*100:5.1f}%)")
                            if avg_optim > 0:
                                self.print(f"     Optimizer Step:         {avg_optim*1000:7.2f}ms ({avg_optim/avg_total*100:5.1f}%)")

                            self.print(f"     ─────────────────────────────────────────")
                            self.print(f"     Total:                  {avg_total*1000:7.2f}ms (100.0%)")

                            # GPU & Memory info
                            self.print("")
                            self.print(f"  💾 GPU & Memory:")
                            gpu_memory_info = get_detailed_gpu_memory_info()
                            for info_line in gpu_memory_info:
                                self.print(f"     {info_line}")
                            self.print("")

                    # Log to logger (commented out to avoid duplicate console output)
                    # Uncomment if you need WandB logging
                    # log_dict = {
                    #     'loss': loss,
                    #     'learning_rate': current_lr,
                    #     'epoch': current_epoch_float,
                    #     'step_time': avg_step_time,
                    # }
                    # if self.profile_timing:
                    #     log_dict.update({
                    #         'timing/data_loading': data_load_time,
                    #         'timing/total_step': total_step_time,
                    #     })
                    #     log_dict.update({f'timing/{k}': v for k, v in step_timing.items()})
                    # self.logger.log_metrics(log_dict, step=self.global_step, prefix='train')

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
