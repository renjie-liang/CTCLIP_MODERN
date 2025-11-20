# CT-CLIP MODERN

A modern implementation of CT-CLIP for medical image-text contrastive learning.

## Quick Start

The default and recommended way to train is **Multi-GPU on a Single Node**.

```bash
# Using bash script (recommended for development)
bash run_setup/bash/train_single_node_multi_gpu.sh

# Or using accelerate directly
accelerate launch \
    --config_file run_setup/configs/accelerate_single_node.yaml \
    train.py \
    --config configs/base_config.yaml
```

**Default Configuration:**
- **GPUs**: 2 (configurable in `run_setup/configs/accelerate_single_node.yaml`)
- **Batch Size**: 4 per GPU
- **Learning Rate**: 1.25e-6
- **Epochs**: 10
- **Precision**: fp16 mixed precision
- **Config**: `configs/base_config.yaml`

## Project Structure

```
CTCLIP_MODERN/
├── train.py                      # Training entry point
├── inference.py                  # Inference entry point
├── configs/                      # Training configurations
│   ├── base_config.yaml          # Default training config
│   ├── debug_config.yaml         # Debug settings
│   └── experiments/              # Experiment-specific configs
├── run_setup/                    # Training scripts and setup
│   ├── bash/                     # Bash training scripts
│   ├── slurm/                    # SLURM job scripts
│   ├── configs/                  # Accelerate configurations
│   ├── requirements.txt          # Python dependencies
│   └── b200_env.yml              # Conda environment
├── src/                          # Source code
│   ├── models/                   # Model implementations
│   ├── training/                 # Training logic
│   ├── data/                     # Data loaders
│   ├── validation/               # Evaluation
│   ├── metrics/                  # Metrics
│   ├── loggers/                  # Logging utilities
│   └── utils/                    # Utilities
└── scripts/                      # Helper scripts
```

## Training Options

### 1. Single GPU (Quick Testing)

For rapid testing and debugging.

```bash
# Using SLURM
sbatch run_setup/slurm/single_gpu.slurm
```

**Configuration:**
- 1 GPU
- Batch size: 8
- Memory: 200GB
- No distributed training

### 2. Multi-GPU Single Node (Recommended)

For most training scenarios.

```bash
# Using SLURM
sbatch run_setup/slurm/single_node_multi_gpu.slurm

# Or using bash script
bash run_setup/bash/train_single_node_multi_gpu.sh
```

**Configuration:**
- Default: 4 GPUs (configurable)
- Batch size per GPU: 8
- Effective batch size: 32 (8 × 4)
- Memory: 400GB
- Uses Accelerate for distributed training

**To change GPU count:**
1. Edit `run_setup/slurm/single_node_multi_gpu.slurm`: `#SBATCH --gres=gpu:N`
2. Edit `run_setup/configs/accelerate_single_node.yaml`: `num_processes: N`

### 3. Multi-Node Multi-GPU (Large Scale)

For large-scale training across multiple machines.

```bash
sbatch run_setup/slurm/multi_node_multi_gpu.slurm
```

**Configuration:**
- Default: 2 nodes × 4 GPUs = 8 GPUs total
- Uses Accelerate + SLURM for distributed training
- Requires cluster-specific settings (partition, account, QoS)

**To change node/GPU configuration:**
1. Edit `run_setup/slurm/multi_node_multi_gpu.slurm`: `#SBATCH --nodes` and `#SBATCH --gpus-per-node`
2. Edit `run_setup/configs/accelerate_multi_node.yaml`: `num_machines` and `num_processes`

## Configuration

### Base Configuration

Edit `configs/base_config.yaml` to customize:
- **Batch size**: `batch_size: 4`
- **Learning rate**: `learning_rate: 1.25e-6`
- **Epochs**: `max_epochs: 10`
- **Validation**: Every 0.5 epoch with 200 samples
- **Checkpointing**: Keep 3 best models (by AUROC)
- **Models**: CTViT (image) + BiomedBERT (text)

### Accelerate Configuration

- **Single node**: `run_setup/configs/accelerate_single_node.yaml`
  - Set `num_processes` to match your GPU count
  - Default: 2 GPUs with fp16 mixed precision

- **Multi-node**: `run_setup/configs/accelerate_multi_node.yaml`
  - Set `num_machines` and `num_processes`
  - Default: 2 nodes, 8 GPUs total

### Environment Setup

The project uses micromamba for environment management. Before running SLURM scripts, update:
- `MAMBA_EXE`: Path to micromamba executable
- `MAMBA_ROOT_PREFIX`: Micromamba root directory
- Project directory paths

For SLURM clusters, also configure:
- `--partition`: Cluster partition name
- `--account`: Account name
- `--qos`: Quality of Service name

## Output Directories

Training outputs are saved to:
- **SLURM logs**: `out_slurm/`
- **Training logs**: `logs/`
- **Model checkpoints**: `saves/`

## Troubleshooting

If training fails, check:

1. **SLURM error logs**: `out_slurm/train_*.err`
2. **GPU allocation**: Verify `CUDA_VISIBLE_DEVICES` is set correctly
3. **GPU count mismatch**: Ensure Accelerate config matches SLURM GPU request
4. **Memory**: Multi-GPU training requires more memory
5. **Dataset paths**: Verify WebDataset paths are correct
6. **Environment**: Ensure all dependencies are installed (`run_setup/requirements.txt`)

### Common Issues

- **"No GPU available"**: Check SLURM GPU allocation and CUDA setup
- **Out of memory**: Reduce batch size or enable gradient accumulation
- **Accelerate not found**: Install with `pip install accelerate`
- **Config file errors**: Verify YAML syntax and file paths

## License

See LICENSE file for details.
