#!/bin/bash

# Master launch script for all ablation experiments
# Usage:
#   ./launch_all_experiments.sh          # Launch all experiments
#   ./launch_all_experiments.sh baseline # Launch only baseline
#   ./launch_all_experiments.sh ablation # Launch only ablation studies

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "CTCLIP Ablation Study Launcher"
echo "========================================"
echo ""

# Function to submit a job and print info
submit_job() {
    local script_name=$1
    local description=$2

    echo "Submitting: $description"
    echo "  Script: $script_name"

    job_id=$(sbatch "$SCRIPT_DIR/$script_name" | awk '{print $4}')

    if [ -n "$job_id" ]; then
        echo "  Job ID: $job_id"
        echo "  ✓ Submitted successfully"
    else
        echo "  ✗ Failed to submit"
    fi
    echo ""
}

# Parse command line argument
MODE=${1:-all}

case "$MODE" in
    baseline)
        echo "Launching BASELINE experiment only..."
        echo ""
        submit_job "run_baseline.slurm" "Baseline (no optimizations)"
        ;;

    optimized|full)
        echo "Launching FULL OPTIMIZED experiment only..."
        echo ""
        submit_job "run_optimized_full.slurm" "Full Optimized (all optimizations)"
        ;;

    ablation)
        echo "Launching all ABLATION STUDIES (6 experiments)..."
        echo ""
        submit_job "run_ablation_flash_only.slurm" "Ablation: FlashAttention only"
        submit_job "run_ablation_rms_only.slurm" "Ablation: RMSNorm only"
        submit_job "run_ablation_swiglu_only.slurm" "Ablation: SwiGLU only"
        submit_job "run_ablation_flash_rms.slurm" "Ablation: FlashAttention + RMSNorm"
        submit_job "run_ablation_flash_swiglu.slurm" "Ablation: FlashAttention + SwiGLU"
        submit_job "run_ablation_rms_swiglu.slurm" "Ablation: RMSNorm + SwiGLU"
        ;;

    all)
        echo "Launching ALL experiments (8 total)..."
        echo ""
        echo "--- Main Comparisons ---"
        submit_job "run_baseline.slurm" "Baseline (no optimizations)"
        submit_job "run_optimized_full.slurm" "Full Optimized (all optimizations)"
        echo ""
        echo "--- Ablation Studies ---"
        submit_job "run_ablation_flash_only.slurm" "Ablation: FlashAttention only"
        submit_job "run_ablation_rms_only.slurm" "Ablation: RMSNorm only"
        submit_job "run_ablation_swiglu_only.slurm" "Ablation: SwiGLU only"
        submit_job "run_ablation_flash_rms.slurm" "Ablation: FlashAttention + RMSNorm"
        submit_job "run_ablation_flash_swiglu.slurm" "Ablation: FlashAttention + SwiGLU"
        submit_job "run_ablation_rms_swiglu.slurm" "Ablation: RMSNorm + SwiGLU"
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo ""
        echo "Usage:"
        echo "  $0 [all|baseline|optimized|ablation]"
        echo ""
        echo "Modes:"
        echo "  all       - Launch all 8 experiments (default)"
        echo "  baseline  - Launch only baseline experiment"
        echo "  optimized - Launch only full optimized experiment"
        echo "  ablation  - Launch only the 6 ablation studies"
        exit 1
        ;;
esac

echo "========================================"
echo "All jobs submitted!"
echo ""
echo "Check job status with: squeue -u \$USER"
echo "Cancel a job with: scancel <job_id>"
echo "View output logs in: out_slurm/"
echo "========================================"
