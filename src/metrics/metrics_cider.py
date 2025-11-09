# metrics_cider.py
"""
CIDEr (pycocoevalcap) for report-generation evaluation.
Returns both sentence-level and corpus-level metrics (scaled to 0–1).

Notes
-----
- Native CIDEr scores from pycocoevalcap are typically ~0–10 (CIDEr-D range).
  To keep all metrics on a common 0–1 scale, we divide by 10.0.
  If you prefer raw CIDEr, set scale_to_unit=False.
- Case-insensitive: lowercased internally.
- pycocoevalcap expects dicts: {idx: ["hyp"]}, {idx: ["ref"]}.
"""

from typing import List, Dict, Tuple
from pycocoevalcap.cider.cider import Cider
import numpy as np


def compute_cider_batch(
    study_ids: List[str],
    hyps: List[str],
    refs: List[str],
    scale_to_unit: bool = True,  # keep True to output 0–1
) -> Tuple[Dict[str, float], float]:
    """
    Compute CIDEr at sentence and corpus level.

    Parameters
    ----------
    study_ids : list of str
        List of study IDs
    hyps : list of str
        Generated reports
    refs : list of str
        Reference reports
    scale_to_unit : bool
        If True, scale CIDEr scores to 0–1 range by dividing by 10.0

    Returns
    -------
    Tuple[Dict[str, float], float]
        - sentence_dict: study_id -> cider_score (0–1 if scaled, 0–10 otherwise)
        - corpus_score: mean CIDEr score across all samples
    """
    scorer = Cider()
    hyps_dict = {i: [(h or "").strip().lower()] for i, h in enumerate(hyps)}
    refs_dict = {i: [(r or "").strip().lower()] for i, r in enumerate(refs)}

    # scorer.compute_score returns (average_score, individual_scores)
    # average_score: float (corpus-level mean)
    # individual_scores: array of per-sample scores (~0–10)
    avg_score, individual_scores = scorer.compute_score(refs_dict, hyps_dict)
    # Optionally scale to 0–1
    scale = 10.0 if scale_to_unit else 1.0

    # Build sentence-level results with 4 decimal places
    sentence_results: Dict[str, float] = {}
    for idx, sid in enumerate(study_ids):
        sentence_results[sid] = round(float(individual_scores[idx]) / scale, 4)
    
    # Corpus-level score with 4 decimal places
    corpus_score = round(float(avg_score) / scale, 4)
    
    return sentence_results, corpus_score