"""
Step-based CT-CLIP Trainer

Key features:
- Step-based training (not epoch-based)
- Warmup + Cosine LR scheduling
- Single-GPU training with PyTorch AMP
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

        # Info about gradient accumulation
        if self.gradient_accumulation_steps > 1:
            print(f"Using gradient accumulation with {self.gradient_accumulation_steps} steps")
            print(f"Effective batch size: {self.batch_size * self.gradient_accumulation_steps}")

        # Store configs for later use (after dataset is loaded)
        self._training_cfg = training_cfg
        self._validation_cfg = validation_cfg

        self.results_folder = Path(checkpoint_cfg['save_dir'])
        self.results_folder.mkdir(parents=True, exist_ok=True)

        # Device setup (single GPU)
        device_cfg = config.get('device', {})
        if device_cfg.get('use_cuda', True) and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_cfg.get('cuda_device', 0)}")
        else:
            self.device = torch.device('cpu')

        # Mixed precision scaler for AMP
        self.use_amp = True  # Enable automatic mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Tokenizer
        text_cfg = config['model']['text_encoder']
        self.tokenizer = AutoTokenizer.from_pretrained(
            text_cfg['path'],
            do_lower_case=text_cfg['do_lower_case'],
            trust_remote_code=True
        )

        # Model selection metric
        self.best_metric = checkpoint_cfg.get('best_metric', 'weighted_auroc')
        # Validate best_metric
        valid_metrics = ['weighted_auroc', 'macro_auroc']
        if self.best_metric not in valid_metrics:
            raise ValueError(f"Invalid best_metric '{self.best_metric}'. Must be one of {valid_metrics}")
        self.print(f"Model selection metric: {self.best_metric}")

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

        # Move model to device
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

        # Enable model internal profiling
        if self.profile_timing:
            self.model.profile_timing = True
            if hasattr(self.model, 'visual_transformer'):
                self.model.visual_transformer.profile_timing = True
            self.print("✅ Model profiling enabled")

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

        # # Monitor model (WandB)
        # if config['logging'].get('use_wandb', False):
        #     log_freq = config['logging'].get('log_every_n_steps', 1000)
        #     self.logger.watch_model(self.model, log_freq=log_freq)

    def _load_pathology_classes(self, labels_path: str) -> List[str]:
        """Load pathology class names from CSV"""
        df = pd.read_csv(labels_path)
        exclude_cols = {'study_id', 'VolumeName'}
        pathology_cols = [col for col in df.columns if col not in exclude_cols]
        return pathology_cols

    def save_checkpoint(self, metrics: Dict):
        """Save checkpoint"""
        save_path = self.checkpoint_manager.save_checkpoint(
            epoch=self.current_epoch,
            global_step=self.global_step,
            model=self.model,
            optimizer=self.optim,
            scheduler=self.scheduler,
            metrics=metrics,
            config=self.config
        )

        self.print(f'✓ Checkpoint saved: {save_path}')

        # Check if best (using configured best_metric for model selection)
        current_auroc = metrics.get(self.best_metric, 0.0)
        if current_auroc > self.best_auroc:
            self.best_auroc = current_auroc
            self.print(f'✅ New best model! {self.best_metric}: {self.best_auroc:.4f}')

    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """Load checkpoint

        Args:
            path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state (set False if model architecture changed)
        """
        path = Path(path)

        checkpoint = self.checkpoint_manager.load_checkpoint(
            str(path),
            model=self.model,
            optimizer=self.optim,
            scheduler=self.scheduler,
            load_optimizer=load_optimizer
        )

        self.global_step = checkpoint.get('global_step', 0)
        self.current_epoch = checkpoint['epoch']
        self.best_auroc = checkpoint.get('metrics', {}).get(self.best_metric, 0.0)

        self.print(f"✅ Resumed from step {self.global_step}, epoch {self.current_epoch}")

    def print(self, msg):
        """Print message"""
        print(msg)

    @property
    def is_main(self):
        """Check if main process (always True for single GPU)"""
        return True

    def train_step(self, dataloader_iter, batch_idx: int) -> Dict:
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

        if self.profile_timing:
            torch.cuda.synchronize()
        t0 = time.time()

        batch = next(dataloader_iter)

        t1 = time.time()
        timing['data_loading'] = t1 - t0

        volume_tensor, report_text, disease_labels, study_id, embed_tensor = batch
        volume_tensor = volume_tensor.to(device, non_blocking=True)

        t2 = time.time()
        timing['data_to_device'] = t2 - t1

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
        t3 = time.time()
        timing['tokenize'] = t3 - t2

        if self.use_amp:
            with torch.amp.autocast('cuda'):
                loss = self.model(text_tokens, volume_tensor, return_loss=True, device=device)
        else:
            loss = self.model(text_tokens, volume_tensor, return_loss=True, device=device)

        if self.profile_timing:
            torch.cuda.synchronize()
        t4 = time.time()
        timing['forward'] = t4 - t3

        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if self.profile_timing: 
            torch.cuda.synchronize()
        t5 = time.time()
        timing['backward'] = t5 - t4

    # Update weights after accumulation
        if self.use_amp:
            # Unscale before gradient clipping
            if self.max_grad_norm:
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optim.step()

        self.scheduler.step()
        self.optim.zero_grad()


        self.global_step += 1
        self.eta_calculator.update()
        
        t6 = time.time()
        timing['optimizer_step'] = t6 - t5

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

        # Macro metrics
        macro_auroc_star = " ⭐" if self.best_metric == 'macro_auroc' else ""
        self.print(f"Macro AUROC: {results['macro_auroc']:.4f}{macro_auroc_star}")
        if 'macro_auprc' in results:
            self.print(f"Macro AUPRC: {results['macro_auprc']:.4f}")
        if 'macro_f1' in results:
            self.print(f"Macro F1: {results['macro_f1']:.4f}")

        self.print(f"---")

        # Weighted metrics
        weighted_auroc_star = " ⭐" if self.best_metric == 'weighted_auroc' else ""
        self.print(f"Weighted AUROC: {results['weighted_auroc']:.4f}{weighted_auroc_star}")
        if 'weighted_auprc' in results:
            self.print(f"Weighted AUPRC: {results['weighted_auprc']:.4f}")
        if 'weighted_f1' in results:
            self.print(f"Weighted F1: {results['weighted_f1']:.4f}")

        self.print(f"{'='*80}\n")

        # Log metrics
        self.logger.log_metrics(results, step=self.global_step, prefix='val')

        return results
        
    def train(self):
        """Main training loop (step-based, no gradient accumulation)."""
        self.print(f"\n{'='*80}")
        self.print("Starting Training")
        self.print(f"{'='*80}\n")

        epoch = self.current_epoch
        dataloader_iter = iter(self.train_dataloader)
        batch_idx_in_epoch = 0
        # Global batch index for gradient accumulation (never resets)
        global_batch_idx = 0

        log_every = self.config['logging'].get('log_every_n_steps', 10)

        while self.global_step < self.max_steps:
            self.model.train()
            # 整个 step 计时（含数据加载 + GPU 计算）
            if self.profile_timing:
                torch.cuda.synchronize()
            step_start = time.time()

            # 一个 step：内部会 next(dataloader_iter)
            try:
                result = self.train_step(dataloader_iter, batch_idx_in_epoch)
            except StopIteration:
                # 一个 epoch 结束，重置 dataloader
                epoch += 1
                self.current_epoch = epoch
                self.print(f"\n--- Epoch {epoch} ---")
                dataloader_iter = iter(self.train_dataloader)
                batch_idx_in_epoch = 0
                result = self.train_step(dataloader_iter, batch_idx_in_epoch)

            loss = result['loss']
            step_timing = result['timing']
            batch_idx_in_epoch += 1
            global_batch_idx += 1

            if self.profile_timing:
                torch.cuda.synchronize()
            total_step_time = time.time() - step_start

            # Data 阶段时间
            data_load_time = step_timing['data_loading']
            model_time = total_step_time - data_load_time

            # ETA / 进度信息
            avg_step_time = self.eta_calculator.get_avg_step_time()
            eta = self.eta_calculator.get_eta(self.global_step, self.max_steps)
            elapsed = self.eta_calculator.get_elapsed_time()
            current_lr = self.optim.param_groups[0]['lr']
            current_epoch_float = self.global_step / max(1, self.steps_per_epoch)

            # 简单一行 log
            if self.global_step % log_every == 0 or self.global_step == 1:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.print(
                    f"[{timestamp}] Ep {current_epoch_float:.2f} | "
                    f"Step {self.global_step}/{self.max_steps} | "
                    f"loss: {loss:.4f} | lr: {current_lr:.2e} | "
                    f"time: {total_step_time:.2f}s (data: {data_load_time:.2f}s, model: {model_time:.2f}s) | "
                    f"ETA: {eta} | elapsed: {elapsed}"
                )




            # 累积 timing 统计 & 打详细 breakdown
            if self.profile_timing:
                # 假设 train_step 永远填好了这些 key（即使是 0）
                self.timing_stats['total_step'].append(total_step_time)
                self.timing_stats['data_loading'].append(step_timing['data_loading'])
                self.timing_stats['data_to_device'].append(step_timing['data_to_device'])
                self.timing_stats['tokenize'].append(step_timing['tokenize'])
                self.timing_stats['forward'].append(step_timing['forward'])
                self.timing_stats['backward'].append(step_timing['backward'])
                self.timing_stats['optimizer_step'].append(step_timing['optimizer_step'])

                # 每 100 step 打一次详细统计
                if self.global_step > 0 and self.global_step % 100 == 0 and self.timing_stats['total_step']:
                    self.print(f"  📊 Timing Breakdown (avg last 100 steps):")

                    # 最近 100 个 step
                    window = 100
                    avg_total        = float(np.mean(self.timing_stats['total_step'][-window:]))
                    avg_data_load    = float(np.mean(self.timing_stats['data_loading'][-window:]))
                    avg_data_device  = float(np.mean(self.timing_stats['data_to_device'][-window:]))
                    avg_tokenize     = float(np.mean(self.timing_stats['tokenize'][-window:]))
                    avg_forward_ext  = float(np.mean(self.timing_stats['forward'][-window:]))
                    avg_backward     = float(np.mean(self.timing_stats['backward'][-window:]))
                    avg_optim        = float(np.mean(self.timing_stats['optimizer_step'][-window:]))

                    pct = lambda x: (x / avg_total * 100.0) if avg_total > 0 else 0.0

                    # Data Stage
                    self.print("  ⏱ Data Stage:")
                    self.print(f"     ├─ Data Loading:        {avg_data_load*1000:7.2f}ms ({pct(avg_data_load):5.1f}%)")
                    self.print(f"     ├─ Data to Device:      {avg_data_device*1000:7.2f}ms ({pct(avg_data_device):5.1f}%)")
                    self.print(f"     └─ Tokenize:            {avg_tokenize*1000:7.2f}ms ({pct(avg_tokenize):5.1f}%)")
                    self.print("")

                    # Model Forward：外层计时 vs 模型内部计时
                    model_timing   = self.model.timing_buffer
                    internal_total = model_timing.get('total_forward', -1.0)
                    forward_diff   = avg_forward_ext - internal_total

                    self.print("  ⏱ Model Forward Summary:")
                    self.print(f"     ├─ Trainer avg forward: {avg_forward_ext*1000:7.2f}ms")
                    self.print(f"     ├─ Internal total:      {internal_total*1000:7.2f}ms")
                    self.print(f"     └─ Δ(outer - inner):    {forward_diff*1000:7.2f}ms")
                    self.print("")

                    # 下面用 internal_total 做分母算占比
                    forward_total = internal_total
                    pct_f = lambda x: (x / forward_total * 100.0) if forward_total > 0 else 0.0

                    self.print(f"  ⏱ Model Forward Breakdown (internal):")
                    self.print(f"     Total Forward:          {forward_total*1000:7.2f}ms (100.0%)")

                    ssl_time        = model_timing.get('ssl', 0.0)
                    data_aug_time   = model_timing.get('data_augmentation', 0.0)
                    text_enc_time   = model_timing.get('text_encoder', 0.0)
                    text_post_time  = model_timing.get('text_postprocess', 0.0)
                    visual_enc_time = model_timing.get('visual_encoder', 0.0)
                    image_pool_time = model_timing.get('image_pooling', 0.0)
                    embed_sel_time  = model_timing.get('embed_selection', 0.0)
                    proj_time       = model_timing.get('projection', 0.0)
                    similarity_time = model_timing.get('similarity_computation', 0.0)
                    loss_comp_time  = model_timing.get('loss_computation', 0.0)
                    ctvit_detail    = model_timing.get('ctvit_detail', {})

                    used_time = (
                        ssl_time + data_aug_time + text_enc_time + text_post_time +
                        visual_enc_time + image_pool_time + embed_sel_time +
                        proj_time + similarity_time + loss_comp_time
                    )
                    other_time = max(0.0, forward_total - used_time)

                    if ssl_time > 0:
                        self.print(f"     ├─ SSL (MLM/VisSSL):    {ssl_time*1000:7.2f}ms ({pct_f(ssl_time):5.1f}%)")
                    if data_aug_time > 0:
                        self.print(f"     ├─ Data Augmentation:   {data_aug_time*1000:7.2f}ms ({pct_f(data_aug_time):5.1f}%)")
                    if text_enc_time > 0:
                        self.print(f"     ├─ Text Encoder:        {text_enc_time*1000:7.2f}ms ({pct_f(text_enc_time):5.1f}%)  [BiomedVLP-BERT]")
                    if text_post_time > 0:
                        self.print(f"     ├─ Text Postprocess:    {text_post_time*1000:7.2f}ms ({pct_f(text_post_time):5.1f}%)")

                    # Visual encoder + CTViT 细分
                    if visual_enc_time > 0:
                        self.print(f"     ├─ Visual Encoder:      {visual_enc_time*1000:7.2f}ms ({pct_f(visual_enc_time):5.1f}%)  [CT-VIT]")

                        spatial_data  = ctvit_detail.get('spatial_transformer')
                        temporal_data = ctvit_detail.get('temporal_transformer')

                        def _to_total(x):
                            if isinstance(x, dict):
                                return x.get('total', 0.0)
                            return x or 0.0

                        spatial_time  = _to_total(spatial_data)
                        temporal_time = _to_total(temporal_data)

                        if spatial_time > 0:
                            self.print(f"     │   ├─ Spatial Trans:       {spatial_time*1000:7.2f}ms ({(spatial_time/visual_enc_time*100.0):5.1f}%)")
                            if isinstance(spatial_data, dict):
                                attn_time  = spatial_data.get('attention', 0.0)
                                ff_time    = spatial_data.get('feedforward', 0.0)
                                norm_time  = spatial_data.get('norm', 0.0)
                                labels     = spatial_data.get('labels', {})

                                if attn_time > 0:
                                    attn_label = labels.get('attention', 'Attention')
                                    self.print(f"     │   │   ├─ Attention:       {attn_time*1000:7.2f}ms ({(attn_time/spatial_time*100.0):5.1f}%)  [{attn_label}]")
                                if ff_time > 0:
                                    ff_label = labels.get('feedforward', 'FFN')
                                    self.print(f"     │   │   ├─ Feed-Forward:    {ff_time*1000:7.2f}ms ({(ff_time/spatial_time*100.0):5.1f}%)  [{ff_label}]")
                                if norm_time > 0:
                                    norm_label = labels.get('norm', 'Norm')
                                    self.print(f"     │   │   └─ Normalization:   {norm_time*1000:7.2f}ms ({(norm_time/spatial_time*100.0):5.1f}%)  [{norm_label}]")

                        if temporal_time > 0:
                            self.print(f"     │   └─ Temporal Trans:      {temporal_time*1000:7.2f}ms ({(temporal_time/visual_enc_time*100.0):5.1f}%)")
                            if isinstance(temporal_data, dict):
                                attn_time  = temporal_data.get('attention', 0.0)
                                ff_time    = temporal_data.get('feedforward', 0.0)
                                norm_time  = temporal_data.get('norm', 0.0)
                                labels     = temporal_data.get('labels', {})

                                if attn_time > 0:
                                    attn_label = labels.get('attention', 'Attention')
                                    self.print(f"     │       ├─ Attention:       {attn_time*1000:7.2f}ms ({(attn_time/temporal_time*100.0):5.1f}%)  [{attn_label}]")
                                if ff_time > 0:
                                    ff_label = labels.get('feedforward', 'FFN')
                                    self.print(f"     │       ├─ Feed-Forward:    {ff_time*1000:7.2f}ms ({(ff_time/temporal_time*100.0):5.1f}%)  [{ff_label}]")
                                if norm_time > 0:
                                    norm_label = labels.get('norm', 'Norm')
                                    self.print(f"     │       └─ Normalization:   {norm_time*1000:7.2f}ms ({(norm_time/temporal_time*100.0):5.1f}%)  [{norm_label}]")

                    if image_pool_time > 0:
                        self.print(f"     ├─ Image Pooling:       {image_pool_time*1000:7.2f}ms ({pct_f(image_pool_time):5.1f}%)")
                    if embed_sel_time > 0:
                        self.print(f"     ├─ Embed Selection:     {embed_sel_time*1000:7.2f}ms ({pct_f(embed_sel_time):5.1f}%)")
                    if proj_time > 0:
                        self.print(f"     ├─ Projection:          {proj_time*1000:7.2f}ms ({pct_f(proj_time):5.1f}%)")
                    if similarity_time > 0:
                        self.print(f"     ├─ Similarity Compute:  {similarity_time*1000:7.2f}ms ({pct_f(similarity_time):5.1f}%)  [einsum]")
                    if loss_comp_time > 0:
                        self.print(f"     ├─ Loss Computation:    {loss_comp_time*1000:7.2f}ms ({pct_f(loss_comp_time):5.1f}%)")
                    if other_time > 0:
                        self.print(f"     └─ Other ops:           {other_time*1000:7.2f}ms ({pct_f(other_time):5.1f}%)")

                    self.print("")
                    # Backward & Optimizer
                    self.print(f"  ⏱ Backward & Optimizer:")
                    if avg_backward > 0:
                        self.print(f"     ├─ Backward Pass:       {avg_backward*1000:7.2f}ms ({pct(avg_backward):5.1f}%)")
                    if avg_optim > 0:
                        self.print(f"     └─ Optimizer Step:      {avg_optim*1000:7.2f}ms ({pct(avg_optim):5.1f}%)")

                    self.print("  ────────────────────────────────────────────────────────────────")
                    self.print(f"  Total Step:                {avg_total*1000:7.2f}ms (100.0%)")

                    # GPU & Memory
                    self.print("")
                    self.print("  💾 GPU & Memory:")
                    gpu_memory_info = get_detailed_gpu_memory_info()
                    for info_line in gpu_memory_info:
                        self.print(f"     {info_line}")
                    self.print("")

            # Validation & Checkpoint
            if self.global_step > 0 and self.global_step % self.eval_every_n_steps == 0:
                metrics = self.validate()
                self.model.train()
                if metrics:
                    self.save_checkpoint(metrics)
            elif self.global_step > 0 and self.global_step % self.save_every_n_steps == 0:
                metrics = {self.best_metric: self.best_auroc}
                self.save_checkpoint(metrics)

        # 训练结束，最后一次验证 + 保存
        self.print(f"\n{'='*80}")
        self.print("Training Complete!")
        self.print(f"{'='*80}")
        metrics = self.validate()
        if metrics:
            self.save_checkpoint(metrics)

        self.print(f"\nBest {self.best_metric}: {self.best_auroc:.4f}")
        self.print(f"Results saved to: {self.results_folder}\n")

        self.logger.finish()

