# metrics_radgraph.py
"""
RadGraph F1 metric for radiology report evaluation.
Evaluates reports based on radiology knowledge graph entities and relations.

Returns both sentence-level and corpus-level metrics.
"""

from typing import List, Dict, Tuple
from radgraph import F1RadGraph


def compute_radgraph_batch(
    study_ids: List[str],
    hyps: List[str],
    refs: List[str],
    model: F1RadGraph = None,
    reward_level: str = "all",
    model_type: str = "radgraph-xl"
) -> Tuple[Dict[str, float], float]:
    """
    Compute RadGraph F1 at sentence and corpus level.
    
    RadGraph evaluates reports by extracting entities and relations into
    a knowledge graph, then computing graph-based F1 scores.

    Parameters
    ----------
    study_ids : list of str
        List of study IDs
    hyps : list of str
        Generated reports
    refs : list of str
        Reference reports
    model : F1RadGraph, optional
        Pre-initialized F1RadGraph model (recommended for efficiency)
    reward_level : str, default="all"
        Reward level: "all", "simple", or "complete"
        - "all": All entities and relations
        - "simple": Entity matching only
        - "complete": Entities + relations
    model_type : str, default="radgraph-xl"
        Model type: "radgraph-xl" (recommended) or "radgraph"

    Returns
    -------
    Tuple[Dict[str, float], float]
        - sentence_dict: study_id -> radgraph score (0–1)
        - corpus_score: mean RadGraph F1 score (0–1)
        
    Notes
    -----
    - First run will download model weights (~500MB for radgraph-xl)
    - Requires GPU for reasonable performance (works on CPU but very slow)
    - radgraph-xl is more accurate but slower than radgraph
    """
    # Initialize model if not provided
    if model is None:
        print(f"  → Initializing RadGraph model (type={model_type}, reward_level={reward_level})")
        model = F1RadGraph(reward_level=reward_level, model_type=model_type)
        print(f"  ✓ RadGraph model loaded")
    # Call RadGraph
    # Returns: (mean_reward, reward_list, hyp_annotations, ref_annotations)
    mean_reward, reward_list,  hypothesis_annotation_lists, reference_annotation_lists = model(hyps=hyps, refs=refs)
    breakpoint()
    
    # --- Sentence-level results ---
    # reward_list is a list of per-sample F1 scores
    sentence_results: Dict[str, float] = {}
    for idx, sid in enumerate(study_ids):
        sentence_results[sid] = round(float(reward_list[idx]), 4)
    
    # --- Corpus-level result ---
    # mean_reward is already the corpus-level average
    corpus_score = round(float(mean_reward), 4)
    
    return sentence_results, corpus_score
