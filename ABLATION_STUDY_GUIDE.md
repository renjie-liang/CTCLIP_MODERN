# CT-CLIP Optimization Ablation Study - Quick Guide

## Created Files Summary

### Config Files (8 experiments)
```
configs/experiments/
├── baseline.yaml                    # ❌ ❌ ❌ (no optimizations)
├── optimized_full.yaml              # ✅ ✅ ✅ (all optimizations)
├── ablation_flash_only.yaml         # ✅ ❌ ❌
├── ablation_rms_only.yaml           # ❌ ✅ ❌
├── ablation_swiglu_only.yaml        # ❌ ❌ ✅
├── ablation_flash_rms.yaml          # ✅ ✅ ❌
├── ablation_flash_swiglu.yaml       # ✅ ❌ ✅
└── ablation_rms_swiglu.yaml         # ❌ ✅ ✅
```

### SLURM Scripts (8 experiments + 1 launcher)
```
run_setup/slurm/
├── run_baseline.slurm
├── run_optimized_full.slurm
├── run_ablation_flash_only.slurm
├── run_ablation_rms_only.slurm
├── run_ablation_swiglu_only.slurm
├── run_ablation_flash_rms.slurm
├── run_ablation_flash_swiglu.slurm
├── run_ablation_rms_swiglu.slurm
├── launch_all_experiments.sh        # Master launcher
└── README_EXPERIMENTS.md            # Detailed documentation
```

## Quick Launch Commands

### Launch Everything (8 jobs)
```bash
cd run_setup/slurm
./launch_all_experiments.sh all
```

### Launch Specific Groups
```bash
./launch_all_experiments.sh baseline   # Just baseline
./launch_all_experiments.sh optimized  # Just full optimized
./launch_all_experiments.sh ablation   # All 6 ablation studies
```

### Launch Individual Experiments
```bash
# Main comparisons
sbatch run_setup/slurm/run_baseline.slurm
sbatch run_setup/slurm/run_optimized_full.slurm

# Individual optimizations
sbatch run_setup/slurm/run_ablation_flash_only.slurm
sbatch run_setup/slurm/run_ablation_rms_only.slurm
sbatch run_setup/slurm/run_ablation_swiglu_only.slurm

# Pairs of optimizations
sbatch run_setup/slurm/run_ablation_flash_rms.slurm
sbatch run_setup/slurm/run_ablation_flash_swiglu.slurm
sbatch run_setup/slurm/run_ablation_rms_swiglu.slurm
```

## Experiment Matrix

| # | Name | Flash | RMS | SwiGLU | Purpose |
|---|------|-------|-----|--------|---------|
| 1 | baseline | ❌ | ❌ | ❌ | Original performance |
| 2 | optimized_full | ✅ | ✅ | ✅ | Best possible performance |
| 3 | flash_only | ✅ | ❌ | ❌ | Isolate FlashAttention impact |
| 4 | rms_only | ❌ | ✅ | ❌ | Isolate RMSNorm impact |
| 5 | swiglu_only | ❌ | ❌ | ✅ | Isolate SwiGLU impact |
| 6 | flash_rms | ✅ | ✅ | ❌ | Test Flash + RMS synergy |
| 7 | flash_swiglu | ✅ | ❌ | ✅ | Test Flash + SwiGLU synergy |
| 8 | rms_swiglu | ❌ | ✅ | ✅ | Test RMS + SwiGLU synergy |

## What Each Optimization Does

### FlashAttention
- **What**: Fused CUDA kernel for attention computation
- **Expected**: 2-4x speedup, 50% memory reduction
- **Trade-offs**: Requires Ampere+ GPU, slightly different numerics

### RMSNorm
- **What**: Removes mean centering from normalization
- **Expected**: 5-10% speedup vs LayerNorm
- **Trade-offs**: Slightly different training dynamics

### SwiGLU
- **What**: Swish activation instead of GELU in feedforward
- **Expected**: Better model quality (iso-compute)
- **Trade-offs**: None (drop-in replacement)

## Monitoring & Results

### Check Job Status
```bash
squeue -u $USER
```

### View Live Logs
```bash
tail -f out_slurm/baseline_<job_id>.out
tail -f out_slurm/optimized_full_<job_id>.out
```

### Result Locations
- **Checkpoints**: `saves/<experiment_name>/`
- **Logs**: `out_slurm/<experiment_name>_<job_id>.out/err`
- **WandB**: Project `ctclip_ablation`

### Cancel Jobs
```bash
scancel <job_id>        # Cancel specific job
scancel -u $USER        # Cancel all your jobs
```

## Analysis Checklist

After experiments complete, compare:
- [ ] Training throughput (steps/sec)
- [ ] Memory usage (GB per GPU)
- [ ] Training loss curves
- [ ] Validation metrics
- [ ] Time to convergence
- [ ] Final model quality

## Next Steps (After Launching)

1. **Implement the optimizations** in the code (merge branch + refactor)
2. **Test locally** that both baseline and optimized configs work
3. **Launch experiments** using the scripts above
4. **Monitor progress** in WandB and SLURM logs
5. **Analyze results** using the checklist above
6. **Write paper section** with ablation results

## Resource Requirements

Each job uses:
- 2x B200 GPUs
- 64 CPUs
- 200GB RAM
- 72 hour time limit

Total for all 8 experiments: **16 GPUs** running concurrently

## Notes

- All configs inherit from `configs/base_config.yaml`
- Each experiment saves to its own directory (no conflicts)
- WandB tags make it easy to filter experiments
- Scripts include detailed logging for debugging
- See `run_setup/slurm/README_EXPERIMENTS.md` for full documentation

---

**TL;DR**: Run `./run_setup/slurm/launch_all_experiments.sh all` to launch everything!
