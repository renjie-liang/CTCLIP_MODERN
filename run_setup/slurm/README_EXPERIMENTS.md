# CTCLIP Optimization Experiments

This directory contains SLURM scripts for running ablation studies on CT-CLIP model optimizations.

## Experiments Overview

We test three optimizations independently and in combination:
- **FlashAttention**: Fused CUDA kernels for faster attention computation
- **RMSNorm**: Faster normalization (removes mean centering)
- **SwiGLU**: Modern activation function (used in LLaMA)

## Experiment Matrix

| Experiment | FlashAttention | RMSNorm | SwiGLU | Config File | SLURM Script |
|------------|----------------|---------|---------|-------------|--------------|
| **Baseline** | ❌ | ❌ | ❌ | `configs/experiments/baseline.yaml` | `run_baseline.slurm` |
| **Full Optimized** | ✅ | ✅ | ✅ | `configs/experiments/optimized_full.yaml` | `run_optimized_full.slurm` |
| Flash Only | ✅ | ❌ | ❌ | `configs/experiments/ablation_flash_only.yaml` | `run_ablation_flash_only.slurm` |
| RMS Only | ❌ | ✅ | ❌ | `configs/experiments/ablation_rms_only.yaml` | `run_ablation_rms_only.slurm` |
| SwiGLU Only | ❌ | ❌ | ✅ | `configs/experiments/ablation_swiglu_only.yaml` | `run_ablation_swiglu_only.slurm` |
| Flash + RMS | ✅ | ✅ | ❌ | `configs/experiments/ablation_flash_rms.yaml` | `run_ablation_flash_rms.slurm` |
| Flash + SwiGLU | ✅ | ❌ | ✅ | `configs/experiments/ablation_flash_swiglu.yaml` | `run_ablation_flash_swiglu.slurm` |
| RMS + SwiGLU | ❌ | ✅ | ✅ | `configs/experiments/ablation_rms_swiglu.yaml` | `run_ablation_rms_swiglu.slurm` |

**Total: 8 experiments** (2 main + 6 ablations)

## Quick Start

### Launch All Experiments (8 jobs)
```bash
cd run_setup/slurm
chmod +x launch_all_experiments.sh
./launch_all_experiments.sh all
```

### Launch Specific Groups
```bash
# Only baseline
./launch_all_experiments.sh baseline

# Only full optimized
./launch_all_experiments.sh optimized

# Only ablation studies (6 experiments)
./launch_all_experiments.sh ablation
```

### Launch Individual Experiments
```bash
# Baseline
sbatch run_baseline.slurm

# Full optimized
sbatch run_optimized_full.slurm

# Specific ablation
sbatch run_ablation_flash_only.slurm
sbatch run_ablation_rms_only.slurm
sbatch run_ablation_swiglu_only.slurm
# ... etc
```

## Resource Configuration

All experiments use:
- **GPUs**: 2x B200 (configurable in each .slurm file)
- **CPUs**: 64 cores per task
- **Memory**: 200GB
- **Time limit**: 72 hours
- **Partition**: hpg-b200
- **Account/QOS**: xujie

To modify resources, edit the `#SBATCH` directives in each `.slurm` file.

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Check specific job
squeue -j <job_id>

# View output logs
tail -f out_slurm/baseline_<job_id>.out
tail -f out_slurm/optimized_full_<job_id>.out

# Cancel a job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER
```

## Output Locations

Each experiment saves to its own directory:
- **Logs**: `out_slurm/<experiment_name>_<job_id>.out/err`
- **Checkpoints**: `saves/<experiment_name>/`
- **WandB project**: `ctclip_ablation`

Example:
```
saves/
├── baseline/
├── optimized_full/
├── ablation_flash_only/
├── ablation_rms_only/
├── ablation_swiglu_only/
├── ablation_flash_rms/
├── ablation_flash_swiglu/
└── ablation_rms_swiglu/
```

## WandB Integration

All experiments log to WandB project `ctclip_ablation` with appropriate tags:
- Baseline: `["baseline", "no_optimization"]`
- Full Optimized: `["optimized", "flash_attention", "rms_norm", "swiglu"]`
- Ablations: `["ablation", <enabled_optimizations>]`

## Analyzing Results

After all experiments complete, compare:
1. **Training speed** (steps/sec, time per epoch)
2. **Memory usage** (GPU memory, peak allocation)
3. **Model performance** (loss, validation metrics)
4. **Convergence** (loss curves in WandB)

## Expected Benefits

Based on literature:
- **FlashAttention**: 2-4x speedup, 50% memory reduction
- **RMSNorm**: 5-10% speedup vs LayerNorm
- **SwiGLU**: Slight accuracy improvement (iso-compute)

## Troubleshooting

### Job fails immediately
- Check `out_slurm/<experiment>_<job_id>.err` for errors
- Verify config file exists and is valid
- Check GPU availability: `sinfo -p hpg-b200`

### Out of memory
- Reduce batch size in config files
- Enable gradient checkpointing
- Use fewer GPUs (edit `#SBATCH --gres=gpu:N`)

### Flash Attention not available
- Ensure flash-attn is installed: `pip list | grep flash`
- Check CUDA version compatibility
- FlashAttention requires Ampere+ GPUs (B200 is supported)

## Customizing Experiments

To create custom ablation combinations:

1. **Create config file** in `configs/experiments/`:
```yaml
_base_: ../base_config.yaml

experiment:
  name: "my_custom_experiment"

model:
  image_encoder:
    use_flash_attention: true  # your choice
    use_rms_norm: false        # your choice
    use_swiglu: true           # your choice

checkpoint:
  save_dir: "saves/my_custom_experiment"

logging:
  wandb_run_name: "my_custom"
  wandb_tags: ["custom", "flash_attention", "swiglu"]
```

2. **Create SLURM script** in `run_setup/slurm/`:
```bash
#!/bin/bash
#SBATCH --job-name=my_custom
# ... copy from existing script and modify ...

accelerate launch \
    --config_file run_setup/configs/accelerate_single_node.yaml \
    train.py \
    --config configs/experiments/my_custom_experiment.yaml
```

3. **Launch**:
```bash
sbatch run_setup/slurm/my_custom_script.slurm
```

## Notes

- All experiments use multi-GPU training with Accelerate
- Make sure you've merged the FlashAttention_RMSNorm_SwiGLU branch first
- Baseline config maintains backward compatibility with existing code
- Each experiment is independent and can be run in parallel
