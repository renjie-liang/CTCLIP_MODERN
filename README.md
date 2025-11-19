# CT-CLIP Modern Training Pipeline

Refactored CT-CLIP training pipeline with epoch-based training, preprocessed WebDataset loading, and multi-GPU support.

## Key Features

- **Epoch-based training** with automatic step calculation
- **Preprocessed WebDataset** - no CPU preprocessing during training (~50-100ms/sample)
- **Multi-GPU support** - single-node and multi-node (via Accelerate)
- **Warmup + Cosine LR scheduling**
- **Flexible evaluation** - configure by epochs or steps
- **Clean, research-focused code**

## Quick Start

### Single GPU Training
```bash
python train.py --config configs/base_config.yaml
```

### Multi-GPU Training (2 GPUs)
```bash
bash scripts/train_multi_gpu.sh
```

### Debug Mode (Fast testing)
```bash
python train.py --config configs/debug_config.yaml
```

---

## Training Settings

All settings in `configs/base_config.yaml`:

### Basic Configuration
```yaml
training:
  max_epochs: 20              # Train for 20 epochs
  learning_rate: 1.25e-6      # Initial learning rate
  warmup_steps: 1000          # Warmup for 1000 steps (~0.14 epochs)

validation:
  eval_every_n_epochs: 0.5    # Evaluate every 0.5 epoch
  eval_samples: 200           # Validate on 200 samples (for speed)

data:
  batch_size: 4               # Per-GPU batch size
  num_workers: 4              # DataLoader workers
```

### Key Points

**Epoch vs Steps:**
- Config uses **epochs** (easier to understand)
- Training loop uses **steps** internally
- With 29,500 samples and batch_size=4: 1 epoch ≈ 7,375 steps

**Multi-GPU:**
- With 2 GPUs: effective batch size = 8, steps per epoch = 3,687
- With 4 GPUs: effective batch size = 16, steps per epoch = 1,843
- Same epochs → same data coverage regardless of GPU count

**Data Loading:**
- Uses preprocessed WebDataset (no CPU preprocessing)
- Fast loading: ~50-100ms per sample
- Data already normalized, resized, cropped to (480, 480, 240)

---

## Multi-GPU Training

### Single-Node Multi-GPU (Recommended)

**Setup (one-time):**
```bash
# Edit accelerate_config_single_node.yaml
# Change num_processes to your GPU count (2, 4, 8, etc.)
num_processes: 2  # for 2 GPUs
```

**Train:**
```bash
bash scripts/train_multi_gpu.sh
```

### Multi-Node Multi-GPU (SLURM)

**Edit SLURM script:**
```bash
# Edit scripts/train_slurm_multi_node.sh
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
```

**Submit job:**
```bash
sbatch scripts/train_slurm_multi_node.sh
```

**See `scripts/MULTI_GPU_GUIDE.md` for detailed instructions.**

---

## Project Structure

```
CTCLIP_MODERN/
├── train.py                           # Training entry point
├── inference.py                       # Inference with bootstrap CI
├── configs/
│   ├── base_config.yaml              # Main training config (20 epochs)
│   └── debug_config.yaml             # Fast testing (100 steps)
├── accelerate_config_single_node.yaml # Single-node multi-GPU config
├── accelerate_config_multi_node.yaml  # Multi-node config
├── scripts/
│   ├── train_multi_gpu.sh            # Single-node multi-GPU launcher
│   ├── train_slurm_multi_node.sh     # SLURM multi-node launcher
│   └── MULTI_GPU_GUIDE.md            # Detailed multi-GPU guide
└── src/
    ├── models/                        # CTViT and CT-CLIP
    ├── data/                          # WebDataset loader
    ├── training/                      # Trainer, optimizer, scheduler
    ├── validation/                    # Metrics and evaluation
    ├── checkpoint/                    # Checkpoint management
    └── utils/                         # Config, seed, ETA calculator
```

---

## Resume Training

```bash
python train.py \
  --config configs/base_config.yaml \
  --resume saves/checkpoint_step_5000.pt
```

---

## Inference

```bash
# Basic inference
python inference.py \
  --checkpoint saves/best_model.pt \
  --config configs/base_config.yaml \
  --output results/results.json

# With bootstrap confidence intervals (1000 samples)
python inference.py \
  --checkpoint saves/best_model.pt \
  --config configs/base_config.yaml \
  --bootstrap \
  --n_bootstrap 1000
```

---

## Training Progress

You'll see detailed progress during training:

```
Using epoch-based training: 20 epochs = 147500 steps
  Dataset: 29500 samples / batch_size 4 = 7375 steps/epoch

Training duration: 20.00 epochs = 147500 steps
Warmup: 1000 steps (0.14 epochs)
Eval every: 3687 steps (0.50 epochs)
Save every: 3687 steps (0.50 epochs)

Step 5234/147500 (Epoch 0.71) | Loss: 0.1234 | LR: 1.2e-6 |
Time/Step: 0.52s | ETA: 20h 34m | Elapsed: 45m 12s
  GPU: 12.3GB / 80.0GB | Util: 76%
```

---

## Validation & Checkpointing

**During Training:**
- Validate every 0.5 epoch on 200 samples (fast)
- Save checkpoint every 0.5 epoch
- Keep best model based on AUROC
- Keep last 3 checkpoints

**Checkpoint Contains:**
- Model weights
- Optimizer state
- Scheduler state
- Global step, epoch
- Validation metrics

---

## Logging

Supports multiple backends (configure in config file):

- **WandB**: Online experiment tracking
- **TensorBoard**: Local visualization
- **Console**: Terminal output

Enable in `configs/base_config.yaml`:
```yaml
logging:
  use_wandb: true
  use_tensorboard: false
  use_console: true
```

---

## Performance

**Single GPU (batch_size=4):**
- Data loading: ~50-100ms/sample
- Training: ~0.5s/step
- 1 epoch ≈ 1 hour

**2 GPUs (batch_size=4 each, total=8):**
- Steps per epoch: 50% of single GPU
- Training: ~0.3s/step
- 1 epoch ≈ 30 minutes (~2x faster)

**4 GPUs (batch_size=4 each, total=16):**
- Steps per epoch: 25% of single GPU
- Training: ~0.15s/step
- 1 epoch ≈ 15 minutes (~4x faster)

---

## Important Notes

1. **Preprocessed data required** - use `train_preprocessed_webdataset/` shards
2. **Random seed**: All seeds set to 2025 for reproducibility
3. **Multi-GPU**: Learning rate scaling optional (test original LR first)
4. **Gradient stride warning**: Can be safely ignored (performance impact <5%)

---

## Troubleshooting

**Out of memory:**
```yaml
data:
  batch_size: 2  # Reduce from 4 to 2
```

**Multi-GPU not working:**
```bash
# Check GPUs visible
nvidia-smi

# Test with debug config first
bash scripts/train_multi_gpu.sh
```

**Slow data loading:**
- Ensure using preprocessed shards (not raw data)
- Check `num_workers` (4-8 recommended)

---

## Questions?

- Multi-GPU guide: `scripts/MULTI_GPU_GUIDE.md`
- Code is clean and documented - read it!
