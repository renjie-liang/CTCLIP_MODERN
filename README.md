# CT-CLIP Modern Training Pipeline

Refactored CT-CLIP training pipeline with step-based training, warmup scheduling, and multi-GPU support.

## Key Features

- **Step-based training** (instead of epoch-based)
- **Warmup + Cosine LR scheduling**
- **Multi-GPU support** via Accelerator
- **Time estimation** and progress tracking
- **Partial validation** for faster iterations (200 samples)
- **Clean, research-focused code** - fail fast, no robust error handling
- **All random seeds set to 2025**

## Project Structure

```
CTCLIP_MODERN/
├── train.py                    # Training entry point
├── inference.py                # Inference script with bootstrap CI
├── configs/                    # Configuration files
│   ├── base_config.yaml
│   └── experiments/
├── src/
│   ├── models/
│   │   ├── ctvit/             # CT Vision Transformer
│   │   └── ct_clip/           # CT-CLIP model
│   ├── data/                   # Data loading
│   ├── training/               # Training components
│   │   ├── trainer.py         # Step-based Trainer
│   │   ├── optimizer.py       # Optimizer with weight decay grouping
│   │   └── scheduler.py       # Warmup + Cosine scheduler
│   ├── validation/             # Validation and metrics
│   ├── checkpoint/             # Checkpoint management
│   ├── loggers/                # Multi-backend logging (WandB, TensorBoard, Console)
│   └── utils/                  # Utilities (seed, config, time estimation)
└── saves/                      # Saved checkpoints
```

## Configuration

All settings in `configs/base_config.yaml`:

### Training Settings
- `max_steps: 10000` - Total training steps
- `learning_rate: 1.25e-6` - Initial learning rate
- `warmup_steps: 1000` - Linear warmup steps
- `min_lr_ratio: 0.01` - Minimum LR as ratio of initial LR
- `gradient_accumulation_steps: 1` - Gradient accumulation
- `save_every_n_steps: 1000` - Checkpoint save frequency

### Validation Settings
- `eval_every_n_steps: 1000` - Validation frequency
- `eval_samples: 200` - Number of samples to validate (for speed)
- `metrics: ["auroc", "auprc", "f1", "precision", "recall"]`
- `best_metric: "auroc"` - Metric for best model selection

### Data Paths
- `DATA_ROOT: /orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset`
- `WEIGHTS_ROOT: /orange/xujie/liang.renjie/DATA/weights`

## Usage

### Single GPU Training

```bash
python train.py --config configs/base_config.yaml
```

### Multi-GPU Training

```bash
# Using accelerate (simplest way)
accelerate launch --multi_gpu --num_processes 4 train.py --config configs/base_config.yaml
```

Or create accelerate config:

```bash
accelerate config
accelerate launch train.py --config configs/base_config.yaml
```

### Resume Training

```bash
python train.py --config configs/base_config.yaml --resume saves/checkpoint_step_5000.pt
```

### Inference

```bash
# Basic inference
python inference.py --checkpoint saves/best_model.pt --config configs/base_config.yaml --output results/results.json

# With bootstrap confidence intervals
python inference.py --checkpoint saves/best_model.pt --config configs/base_config.yaml --bootstrap --n_bootstrap 1000
```

## Training Progress Display

During training, you'll see:

```
Step 5234/10000 (Epoch 12.8) | Loss: 0.1234 | LR: 1.2e-6 |
Time/Step: 1.23s | ETA: 1h 34m 12s | Elapsed: 2h 15m 30s
```

- **Step**: Current step / Total steps
- **Epoch**: Calculated as current_step / steps_per_epoch
- **Time/Step**: Average time per step (moving average over 100 steps)
- **ETA**: Estimated time remaining
- **Elapsed**: Total elapsed time

## Validation

- During training: validate on 200 samples every 1000 steps (fast)
- During inference: validate on all samples with optional bootstrap CI (thorough)

## Checkpointing

Checkpoints saved to `saves/`:
- Regular checkpoints every 1000 steps
- Best model based on AUROC
- Keeps last 3 checkpoints to save space

Checkpoint contains:
- Model weights
- Optimizer state
- Scheduler state
- Training step and epoch
- Metrics

## Logging

Supports multiple backends:
- **WandB**: Online experiment tracking
- **TensorBoard**: Local visualization
- **Console**: Terminal output

Configure in `configs/base_config.yaml`.

## Important Notes

1. **Random seed**: All seeds set to 2025 for reproducibility
2. **Fail fast**: No extensive error handling - errors raised immediately
3. **Research code**: Optimized for iteration speed, not production robustness
4. **Optimizer**: Uses custom weight decay grouping (better than default PyTorch)
5. **Accelerator**: Simplest multi-GPU support - keep it

## Configuration Changes from Original

| Original | New |
|----------|-----|
| Epoch-based training | Step-based training |
| `num_epochs: 100` | `max_steps: 10000` |
| `every_n_epochs: 1` | `eval_every_n_steps: 1000` |
| `save_every_n_epochs: 5` | `save_every_n_steps: 1000` |
| Full validation | Partial validation (200 samples) |
| No warmup | Warmup + Cosine scheduling |
| Seed: 42 | Seed: 2025 |

## Questions?

Read the code - it's clean and well-structured!
