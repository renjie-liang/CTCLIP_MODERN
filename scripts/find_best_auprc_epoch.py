#!/usr/bin/env python3
"""
Script to find the epoch with highest AUPRC from wandb logs.

This script can work in two modes:
1. Online mode: Uses wandb API to fetch run history from wandb servers
2. Offline mode: Reads wandb summary files from local ./wandb directory
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import wandb
    from wandb.apis.public import Api
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb package not available")


def get_wandb_run_summary(run_path: Path) -> Optional[Dict]:
    """
    Read wandb summary from local files.

    Args:
        run_path: Path to wandb run directory

    Returns:
        Summary dictionary or None
    """
    summary_file = run_path / "files" / "wandb-summary.json"
    if not summary_file.exists():
        return None

    with open(summary_file, 'r') as f:
        return json.load(f)


def get_run_config(run_path: Path) -> Optional[Dict]:
    """
    Read wandb config from local files.

    Args:
        run_path: Path to wandb run directory

    Returns:
        Config dictionary or None
    """
    config_file = run_path / "files" / "config.yaml"
    if not config_file.exists():
        return None

    try:
        import yaml
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except ImportError:
        print("Warning: PyYAML not available, skipping config")
        return None


def get_experiment_name(config: Optional[Dict]) -> str:
    """
    Extract experiment name from config.

    Args:
        config: Config dictionary

    Returns:
        Experiment name or 'unknown'
    """
    if config is None:
        return 'unknown'

    try:
        return config.get('experiment', {}).get('value', {}).get('name', 'unknown')
    except (AttributeError, TypeError):
        return 'unknown'


def parse_output_log_for_best_epoch(run_path: Path, metric_name: str = "val/weighted_auprc") -> Optional[Dict]:
    """
    Parse output.log to find the best epoch for a given metric.

    Args:
        run_path: Path to wandb run directory
        metric_name: Metric to track (e.g., "val/weighted_auprc")

    Returns:
        Dictionary with best epoch info or None
    """
    output_log = run_path / "files" / "output.log"
    if not output_log.exists():
        return None

    # Extract metric short name (e.g., "weighted_auprc" from "val/weighted_auprc")
    metric_short = metric_name.replace('val/', '')

    best_epoch = -1
    best_value = -1
    best_metrics = {}

    try:
        with open(output_log, 'r') as f:
            current_step = None
            current_epoch = None

            for line in f:
                # Match validation step line: "Running Validation (Step 932, Epoch 1.00)"
                if "Running Validation" in line and "Step" in line and "Epoch" in line:
                    import re
                    match = re.search(r'Step (\d+), Epoch ([\d.]+)', line)
                    if match:
                        current_step = int(match.group(1))
                        current_epoch = float(match.group(2))

                # Match the log line with all metrics
                # Format: [timestamp] Step 932 - val | macro_auroc: 0.5808, ..., weighted_auprc: 0.3090, ...
                if current_epoch is not None and " - val |" in line:
                    # Parse all metrics from the line
                    metrics = {}
                    parts = line.split(" - val |")
                    if len(parts) > 1:
                        metric_str = parts[1].strip()
                        # Split by comma and parse each metric
                        for metric_pair in metric_str.split(','):
                            metric_pair = metric_pair.strip()
                            if ':' in metric_pair:
                                key, value = metric_pair.split(':', 1)
                                key = key.strip()
                                try:
                                    value = float(value.strip())
                                    metrics[f'val/{key}'] = value
                                except ValueError:
                                    continue

                    # Check if this epoch has better performance
                    if metric_name in metrics:
                        value = metrics[metric_name]
                        if value > best_value:
                            best_value = value
                            best_epoch = current_epoch
                            best_metrics = metrics

    except Exception as e:
        print(f"Error parsing {output_log}: {e}")
        return None

    if best_epoch < 0:
        return None

    return {
        'best_epoch': int(best_epoch),
        'best_value': best_value,
        'metrics': best_metrics
    }


def find_best_from_online_api(project: str,
                              entity: Optional[str] = None,
                              metric_name: str = "val/weighted_auprc",
                              run_id: Optional[str] = None) -> Tuple[str, int, float, Dict]:
    """
    Find best epoch using wandb online API.

    Args:
        project: W&B project name
        entity: W&B entity/username
        metric_name: Metric to track
        run_id: Specific run ID to analyze

    Returns:
        Tuple of (run_id, best_epoch, best_auprc, best_record)
    """
    if not WANDB_AVAILABLE:
        raise RuntimeError("wandb package is required for online API mode")

    api = Api()

    if entity:
        path = f"{entity}/{project}"
    else:
        path = project

    if run_id:
        # Get specific run
        run = api.run(f"{path}/{run_id}")
        runs = [run]
    else:
        # Get all runs
        runs = api.runs(path)

    best_overall = None
    best_overall_auprc = -1
    best_overall_run_id = None
    best_overall_epoch = -1

    for run in runs:
        print(f"\nAnalyzing run: {run.id} ({run.name})")

        # Get history
        history = run.history()

        if history.empty:
            print("  No history data")
            continue

        # Check if metric exists
        if metric_name not in history.columns:
            print(f"  Metric '{metric_name}' not found")
            continue

        # Find best epoch
        best_idx = history[metric_name].idxmax()
        best_record = history.loc[best_idx]
        best_auprc = best_record[metric_name]
        best_epoch = best_record.get('_step', best_record.get('epoch', best_idx))

        print(f"  Best: Epoch {best_epoch}, {metric_name}: {best_auprc:.4f}")

        if best_auprc > best_overall_auprc:
            best_overall_auprc = best_auprc
            best_overall_run_id = run.id
            best_overall_epoch = int(best_epoch)
            best_overall = best_record.to_dict()

    if best_overall is None:
        raise ValueError(f"No runs found with metric '{metric_name}'")

    return best_overall_run_id, best_overall_epoch, best_overall_auprc, best_overall


def find_best_from_local_files(wandb_dir: str = "./wandb",
                               metric_name: str = "val/weighted_auprc",
                               run_id: Optional[str] = None) -> Tuple[str, int, float]:
    """
    Find best epoch from local wandb files (offline mode).

    Note: This only reads the final summary, not the full history.
    For full history analysis, use online API mode.

    Args:
        wandb_dir: Path to wandb directory
        metric_name: Metric to track
        run_id: Specific run ID to analyze

    Returns:
        Tuple of (run_name, best_epoch, best_auprc)
    """
    wandb_path = Path(wandb_dir)

    if not wandb_path.exists():
        raise ValueError(f"Wandb directory not found: {wandb_dir}")

    # Find run directory
    if run_id:
        run_dirs = list(wandb_path.glob(f"run-*-{run_id}"))
        if not run_dirs:
            raise ValueError(f"Run ID not found: {run_id}")
        run_dirs_to_check = run_dirs
    else:
        # Check all runs
        run_dirs_to_check = sorted(
            [d for d in wandb_path.glob("run-*") if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

    if not run_dirs_to_check:
        raise ValueError("No run directories found")

    best_run_name = None
    best_epoch = -1
    best_auprc = -1
    best_summary = None

    print("\nScanning local wandb runs...")
    print("=" * 60)

    for run_path in run_dirs_to_check:
        summary = get_wandb_run_summary(run_path)

        if summary is None:
            print(f"{run_path.name}: No summary file")
            continue

        if metric_name not in summary:
            print(f"{run_path.name}: Metric '{metric_name}' not found")
            continue

        auprc = summary[metric_name]
        step = summary.get('_step', summary.get('epoch', 0))

        print(f"{run_path.name}: Step {step}, {metric_name}: {auprc:.4f}")

        if auprc > best_auprc:
            best_auprc = auprc
            best_epoch = step
            best_run_name = run_path.name
            best_summary = summary

    if best_run_name is None:
        raise ValueError(f"No runs found with metric '{metric_name}'")

    print("\n" + "=" * 60)
    print(f"BEST RUN: {best_run_name}")
    print(f"Best Epoch/Step: {best_epoch}")
    print(f"Best {metric_name}: {best_auprc:.4f}")
    print("=" * 60)

    # Print other relevant metrics
    if best_summary:
        print("\nOther metrics at this epoch:")
        relevant_metrics = [
            'val/macro_auroc',
            'val/weighted_auprc',
            'val/weighted_auroc',
            'val/macro_f1',
            'val/weighted_f1',
            'val/macro_recall',
            'val/macro_precision',
            'val/weighted_recall',
            'val/weighted_precision'
        ]
        for metric in relevant_metrics:
            if metric in best_summary and metric != metric_name:
                print(f"  {metric}: {best_summary[metric]:.4f}")

    return best_run_name, best_epoch, best_auprc


def analyze_all_local_runs(wandb_dir: str = "./wandb",
                           metric_name: str = "val/weighted_auprc") -> List[Dict]:
    """
    Analyze all local runs and find best epoch for each.

    Args:
        wandb_dir: Path to wandb directory
        metric_name: Metric to track

    Returns:
        List of dictionaries containing run information
    """
    wandb_path = Path(wandb_dir)
    run_dirs = sorted([d for d in wandb_path.glob("run-*") if d.is_dir()],
                      key=lambda x: x.name)

    results = []

    # Define all weighted metrics to extract
    weighted_metrics = [
        'val/weighted_auprc',
        'val/weighted_auroc',
        'val/weighted_f1',
        'val/weighted_precision',
        'val/weighted_recall'
    ]

    for run_path in run_dirs:
        # Get config to extract experiment name
        config = get_run_config(run_path)
        experiment_name = get_experiment_name(config)

        # Parse output.log to find best epoch
        best_info = parse_output_log_for_best_epoch(run_path, metric_name)

        if best_info is None:
            continue

        # Extract all weighted metrics from best epoch
        metrics = {}
        for metric in weighted_metrics:
            if metric in best_info['metrics']:
                metrics[metric] = best_info['metrics'][metric]

        results.append({
            'run_name': run_path.name,
            'experiment_name': experiment_name,
            'best_epoch': best_info['best_epoch'],
            'primary_metric': best_info['best_value'],
            'metrics': metrics
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Find epoch with highest AUPRC from wandb logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find best epoch from local files (offline mode)
  python find_best_auprc_epoch.py

  # Analyze all local runs
  python find_best_auprc_epoch.py --all

  # Use custom metric
  python find_best_auprc_epoch.py --metric val/weighted_auprc

  # Analyze specific run
  python find_best_auprc_epoch.py --run-id e6v1zxzz

  # Use online API (requires wandb login)
  python find_best_auprc_epoch.py --online --project my-project --entity my-username
        """
    )
    parser.add_argument(
        "--wandb-dir",
        default="./wandb",
        help="Path to wandb directory (default: ./wandb)"
    )
    parser.add_argument(
        "--metric",
        default="val/weighted_auprc",
        help="Metric name to track (default: val/weighted_auprc)"
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Specific run ID to analyze (default: all runs)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show summary for all runs"
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Use wandb online API (requires login)"
    )
    parser.add_argument(
        "--project",
        help="W&B project name (required for online mode)"
    )
    parser.add_argument(
        "--entity",
        help="W&B entity/username (for online mode)"
    )

    args = parser.parse_args()

    try:
        if args.online:
            # Online API mode
            if not args.project:
                print("Error: --project is required for online mode")
                return 1

            print(f"Using wandb online API for project: {args.project}")
            run_id, epoch, auprc, record = find_best_from_online_api(
                args.project,
                args.entity,
                args.metric,
                args.run_id
            )
            print(f"\nBest run: {run_id}")
            print(f"Best epoch: {epoch}")
            print(f"Best {args.metric}: {auprc:.4f}")

        else:
            # Local file mode
            print("Using local wandb files (offline mode)")
            print("Note: Only reading final summaries, not full history")
            print()

            if args.all:
                results = analyze_all_local_runs(args.wandb_dir, args.metric)

                if results:
                    # Sort by primary metric (descending)
                    sorted_results = sorted(results, key=lambda x: x['primary_metric'], reverse=True)

                    print("\nAll runs sorted by performance:")
                    print("=" * 140)

                    # Print header with all weighted metrics
                    header = f"{'Experiment Name':<25} {'Best Epoch':<12} {'AUPRC':>8} {'AUROC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}"
                    print(header)
                    print("-" * 140)

                    # Print each run with all weighted metrics
                    for result in sorted_results:
                        exp_name = result['experiment_name']
                        epoch = result['best_epoch']
                        metrics = result['metrics']

                        auprc = metrics.get('val/weighted_auprc', 0)
                        auroc = metrics.get('val/weighted_auroc', 0)
                        f1 = metrics.get('val/weighted_f1', 0)
                        precision = metrics.get('val/weighted_precision', 0)
                        recall = metrics.get('val/weighted_recall', 0)

                        row = f"{exp_name:<25} {epoch:<12} {auprc:>8.4f} {auroc:>8.4f} {f1:>8.4f} {precision:>10.4f} {recall:>8.4f}"
                        print(row)

                    print("=" * 140)
                    print(f"\nSorted by: {args.metric}")
                else:
                    print("No runs found with the specified metric")
            else:
                run_name, epoch, auprc = find_best_from_local_files(
                    args.wandb_dir,
                    args.metric,
                    args.run_id
                )

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
