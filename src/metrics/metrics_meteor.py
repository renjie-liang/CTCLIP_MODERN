# metrics_meteor.py
"""
METEOR (pycocoevalcap) for report-generation evaluation.
Returns both sentence-level and corpus-level metrics.

Notes
-----
- Case-insensitive: lowercased internally.
- pycocoevalcap expects dicts: {idx: ["hyp"]}, {idx: ["ref"]}.
"""

from typing import List, Dict, Tuple
from pycocoevalcap.meteor.meteor import Meteor
import numpy as np


def compute_meteor_batch(
    study_ids: List[str],
    hyps: List[str],
    refs: List[str],
) -> Tuple[Dict[str, float], float]:
    """
    Compute METEOR at sentence and corpus level.

    Parameters
    ----------
    study_ids : list of str
        List of study IDs
    hyps : list of str
        Generated reports
    refs : list of str
        Reference reports

    Returns
    -------
    Tuple[Dict[str, float], float]
        - sentence_dict: study_id -> meteor_score (0–1)
        - corpus_score: mean METEOR score across all samples (0–1)
    """
    scorer = Meteor()
    hyps_dict = {i: [(h or "").strip().lower()] for i, h in enumerate(hyps)}
    refs_dict = {i: [(r or "").strip().lower()] for i, r in enumerate(refs)}

    # scorer.compute_score returns (average_score, individual_scores)
    # average_score: float (corpus-level mean)
    # individual_scores: array of per-sample scores
    avg_score, individual_scores = scorer.compute_score(refs_dict, hyps_dict)
    # Build sentence-level results with 4 decimal places
    sentence_results: Dict[str, float] = {}
    for idx, sid in enumerate(study_ids):
        sentence_results[sid] = round(float(individual_scores[idx]), 4)
    
    # Corpus-level score with 4 decimal places
    corpus_score = round(float(avg_score), 4)
    
    return sentence_results, corpus_score