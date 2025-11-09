# metrics_bleu.py
"""
Unified BLEU metric for report-generation evaluation.
Computes:
  - sacreBLEU  (sentence-level, 0–1)
  - COCO BLEU  (BLEU-1/2/3/4, 0–1)

Returns both sentence-level and corpus-level metrics.
"""

from typing import List, Dict, Tuple
import sacrebleu
from pycocoevalcap.bleu.bleu import Bleu
import numpy as np


def compute_bleu_batch(
    study_ids: List[str],
    hyps: List[str],
    refs: List[str],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Compute both sacreBLEU and COCO BLEU-1/2/3/4 at sentence and corpus level.

    Parameters
    ----------
    study_ids : list of str
        List of study IDs (same order as hyps and refs)
    hyps : list of str
        Generated reports
    refs : list of str
        Reference reports

    Returns
    -------
    Tuple[Dict[str, Dict[str, float]], Dict[str, float]]
        - sentence_dict: study_id -> {
            "bleu_sacre": 0.31,
            "bleu_1": 0.56,
            "bleu_2": 0.41,
            "bleu_3": 0.28,
            "bleu_4": 0.21
          }
        - corpus_dict: {
            "bleu_sacre_corpus": 0.35,
            "bleu_1_corpus": 0.58,
            "bleu_2_corpus": 0.43,
            "bleu_3_corpus": 0.30,
            "bleu_4_corpus": 0.22
          }
    """
    # --- Step 1. sacreBLEU (sentence-level) ---
    bleu_sacre_dict: Dict[str, float] = {}
    for sid, hyp, ref in zip(study_ids, hyps, refs):
        hyp_l = (hyp or "").strip().lower()
        ref_l = (ref or "").strip().lower()
        sb = sacrebleu.sentence_bleu(
            hyp_l, [ref_l],
            smooth_method="exp"
        )
        bleu_sacre_dict[sid] = round(float(sb.score) / 100.0, 4)  # 0–1 scale, 4 decimals

    # --- Step 2. sacreBLEU (corpus-level) ---
    hyps_lower = [(h or "").strip().lower() for h in hyps]
    refs_lower = [[(r or "").strip().lower()] for r in refs]
    corpus_bleu_sacre = sacrebleu.corpus_bleu(
        hyps_lower, refs_lower,
        smooth_method="exp"
    )
    corpus_bleu_sacre_score = round(float(corpus_bleu_sacre.score) / 100.0, 4)  # 4 decimals

    # --- Step 3. COCO BLEU-1/2/3/4 (sentence-level) ---
    scorer = Bleu(4)
    hyps_dict = {i: [h.strip().lower()] for i, h in enumerate(hyps)}
    refs_dict = {i: [r.strip().lower()] for i, r in enumerate(refs)}
    
    # scorer.compute_score returns (average_scores, individual_scores)
    # average_scores: tuple of 4 floats (avg BLEU-1, BLEU-2, BLEU-3, BLEU-4)
    # individual_scores: tuple of 4 arrays (per-sample scores for each BLEU-n)
    avg_scores, individual_scores = scorer.compute_score(refs_dict, hyps_dict)
    bleu_keys = ["bleu_1", "bleu_2", "bleu_3", "bleu_4"]

    # --- Step 4. Combine sentence-level results ---
    sentence_results: Dict[str, Dict[str, float]] = {}
    for idx, sid in enumerate(study_ids):
        res = {"bleu_sacre": bleu_sacre_dict.get(sid, 0.0)}
        for i, key in enumerate(bleu_keys):
            # individual_scores[i] is an array of per-sample scores for BLEU-(i+1)
            res[key] = round(float(individual_scores[i][idx]), 4)  # 4 decimals
        sentence_results[sid] = res

    # --- Step 5. Corpus-level results ---
    # Use avg_scores from COCO BLEU (already corpus-level averages)
    corpus_results = {
        "bleu_sacre_corpus": corpus_bleu_sacre_score,  # Already rounded
        "bleu_1_corpus": round(float(avg_scores[0]), 4),
        "bleu_2_corpus": round(float(avg_scores[1]), 4),
        "bleu_3_corpus": round(float(avg_scores[2]), 4),
        "bleu_4_corpus": round(float(avg_scores[3]), 4),
    }

    return sentence_results, corpus_results