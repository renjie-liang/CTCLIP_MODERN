# metrics_chexbert.py
"""
F1-CheXbert metric for radiology report evaluation.
Evaluates clinical entity extraction from chest X-ray reports.

Returns both sentence-level and corpus-level metrics.
"""

import os
import subprocess
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np
from f1chexbert import F1CheXbert


def ensure_chexbert_checkpoint():
    """
    Ensure CheXbert checkpoint exists, download if necessary.
    
    The checkpoint is stored in ~/.cache/chexbert/chexbert.pth
    If not found, downloads from HuggingFace.
    """
    # Determine cache directory (same as f1chexbert package uses)
    cache_dir = Path.home() / ".cache" / "chexbert"
    checkpoint_path = cache_dir / "chexbert.pth"
    
    # Check if checkpoint exists
    if checkpoint_path.exists():
        print(f"  ✓ CheXbert checkpoint found: {checkpoint_path}")
        return
    
    # Create cache directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Download checkpoint
    print(f"  ⚠ CheXbert checkpoint not found. Downloading...")
    print(f"  → Target: {checkpoint_path}")
    
    url = "https://huggingface.co/StanfordAIMI/RRG_scorers/resolve/main/chexbert.pth"
    
    try:
        # Use wget to download
        subprocess.run(
            ["wget", "-P", str(cache_dir), url],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"  ✓ CheXbert checkpoint downloaded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to download checkpoint with wget: {e}")
        print(f"  → Stderr: {e.stderr}")
        print(f"\n  Please manually download:")
        print(f"  wget -P {cache_dir} {url}")
        raise RuntimeError("CheXbert checkpoint download failed")
    except FileNotFoundError:
        print(f"  ✗ wget not found. Please install wget or manually download:")
        print(f"  wget -P {cache_dir} {url}")
        print(f"\n  Or use curl:")
        print(f"  curl -L -o {checkpoint_path} {url}")
        raise RuntimeError("wget not available for checkpoint download")


def compute_chexbert_batch(
    study_ids: List[str],
    hyps: List[str],
    refs: List[str],
    model: F1CheXbert = None,
) -> Tuple[Dict[str, float], float]:
    """
    Compute F1-CheXbert at sentence and corpus level.
    
    F1-CheXbert evaluates clinical entity extraction accuracy by comparing
    the presence/absence of 14 medical conditions between generated and
    reference radiology reports.

    Parameters
    ----------
    study_ids : list of str
        List of study IDs
    hyps : list of str
        Generated reports
    refs : list of str
        Reference reports
    model : F1CheXbert, optional
        Pre-initialized F1CheXbert model (recommended for efficiency)

    Returns
    -------
    Tuple[Dict[str, float], float]
        - sentence_dict: study_id -> f1_chexbert score (per-sample accuracy, 0–1)
        - corpus_score: micro-averaged F1 score across 5 core pathologies (0–1)
        
    Notes
    -----
    - First run will download ~200MB checkpoint to ~/.cache/chexbert/
    - Requires GPU for reasonable performance (works on CPU but slow)
    - Uses 5 core pathologies: Cardiomegaly, Edema, Consolidation, 
      Atelectasis, Pleural Effusion
    """
    # Ensure checkpoint is downloaded
    ensure_chexbert_checkpoint()
    
    # Initialize model if not provided
    if model is None:
        model = F1CheXbert()
    
    # Call F1CheXbert
    # Returns: (accuracy, accuracy_per_sample, class_report_14, class_report_5)
    accuracy, accuracy_per_sample, class_report, class_report_5 = model(
        hyps=hyps,
        refs=refs
    )
    # --- Sentence-level results ---
    # accuracy_per_sample is a numpy array of per-sample accuracies
    sentence_results: Dict[str, float] = {}
    for idx, sid in enumerate(study_ids):
        sentence_results[sid] = round(float(accuracy_per_sample[idx]), 4)
    
    # --- Corpus-level result ---
    # Use micro-averaged F1 from 5 core pathologies (more meaningful than accuracy)
    # This is the standard metric for CheXbert evaluation
    corpus_score = round(float(class_report_5['micro avg']['f1-score']), 4)
    
    breakpoint()
    ### Should we report all metrics?
    return sentence_results, corpus_score
