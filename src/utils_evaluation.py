# evaluation_utils.py
"""
Utility helpers for the generation-evaluation pipeline.
"""

from pathlib import Path
from typing import Dict, List, Any
import json
import warnings
from datetime import datetime


def load_jsonl(filename):
    """Load JSONL file into list of dictionaries."""
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def save_jsonl(filename, data):
    """Save list of dictionaries to JSONL file."""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e) for e in data]))


def build_case_lists(rows: List[Dict[str, Any]]):
    """Extract study IDs, predictions, and references from loaded JSONL rows."""
    study_ids, hyps, refs = [], [], []
    for r in rows:
        sid = r.get("study_id") or r.get("id")
        if not sid:
            continue
        study_ids.append(str(sid))
        hyps.append(r.get("generated_report", "") or "")
        refs.append(r.get("reference_report", "") or "")
    return study_ids, hyps, refs


def merge_metrics(
    res_cases: dict[str, dict[str, any]],
    metric_dict: dict[str, dict[str, any]] | dict[str, float],
    study_ids: list[str] | None = None,
) -> None:
    """
    Merge all key–value pairs from metric_dict[sid] into res_cases[sid].
    Creates new entries if missing.
    Optionally validates that every expected study_id is present.

    Parameters
    ----------
    res_cases : dict[str, dict[str, any]]
        Global results map being built.
    metric_dict : dict[str, dict[str, any]] | dict[str, float]
        Metric results {study_id: {metric_name: value, ...}} or {study_id: value}.
    study_ids : list[str] | None
        Optional list of all expected study IDs for validation.

    Notes
    -----
    - If study_ids is provided, any ID missing from metric_dict will trigger a warning.
    - No hard error: evaluation continues even if missing.
    """

    # --- optional ID validation ---
    if study_ids is not None:
        missing = [sid for sid in study_ids if sid not in metric_dict]
        if missing:
            warnings.warn(
                f"[merge_metrics] {len(missing)} study_ids missing in metric_dict: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}",
                RuntimeWarning,
            )

    # --- main merge logic ---
    for sid, metrics in metric_dict.items():
        if sid not in res_cases:
            res_cases[sid] = {"study_id": sid}
        
        # Handle both dict and scalar values
        if isinstance(metrics, dict):
            res_cases[sid].update(metrics)
        else:
            # If metrics is a scalar, we need a key name - use the calling context
            # This shouldn't happen with our current design, but handle it gracefully
            res_cases[sid]["value"] = metrics


def write_case_results_jsonl(out_path: Path, study_ids: List[str], res_cases: Dict[str, Dict[str, Any]]):
    """Write results to JSONL in the original input order."""
    ordered = [res_cases[sid] for sid in study_ids if sid in res_cases]
    save_jsonl(out_path, ordered)


def write_corpus_results_json(out_path: Path, corpus_results: Dict[str, float], metadata: Dict[str, Any] = None):
    """
    Write corpus-level results to JSON file.
    
    Parameters
    ----------
    out_path : Path
        Output JSON file path
    corpus_results : Dict[str, float]
        Corpus-level metric results
    metadata : Dict[str, Any], optional
        Additional metadata to include (e.g., timestamp, n_samples)
    """
    output = {
        "metadata": metadata or {},
        "corpus_metrics": corpus_results
    }
    
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)


def write_summary_txt(out_path: Path, corpus_results: Dict[str, float], n_samples: int = None):
    """
    Write a human-readable summary of corpus-level metrics.
    
    Parameters
    ----------
    out_path : Path
        Output text file path
    corpus_results : Dict[str, float]
        Corpus-level metric results
    n_samples : int, optional
        Number of samples evaluated
    """
    with open(out_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Report Generation Evaluation Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if n_samples is not None:
            f.write(f"Total samples: {n_samples}\n")
        f.write("\n")
        
        f.write("Corpus-Level Metrics:\n")
        f.write("-" * 80 + "\n")
        
        # Group metrics by type for better readability
        bleu_metrics = {k: v for k, v in corpus_results.items() if "bleu" in k.lower()}
        other_metrics = {k: v for k, v in corpus_results.items() if "bleu" not in k.lower()}
        
        if bleu_metrics:
            f.write("\nBLEU Metrics:\n")
            for metric, value in sorted(bleu_metrics.items()):
                metric_name = metric.replace("_corpus", "").replace("_", "-").upper()
                f.write(f"  {metric_name:20s}: {value:.4f}\n")
        
        if other_metrics:
            f.write("\nOther Metrics:\n")
            for metric, value in sorted(other_metrics.items()):
                metric_name = metric.replace("_corpus", "").replace("_", " ").upper()
                f.write(f"  {metric_name:20s}: {value:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
