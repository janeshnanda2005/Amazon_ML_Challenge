"""
Decision-threshold search.

Critical detail specific to this competition: the metric is F_0.5 averaged
*per Source 1 entity*, with singletons scored too (1.0 for a correctly
empty prediction, 0.0 for any false merge on a true singleton). That means
the right threshold is whatever maximizes *that* macro average - not
whatever maximizes global/micro precision-recall on the pooled set of
pairs. A pair-level threshold search over-weights entities with many
candidate pairs and under-weights singletons, which this competition
penalizes hard. Always search directly against `evaluate.macro_f_beta`.
"""

import numpy as np

from . import config
from .evaluate import macro_f_beta
from .output import predictions_from_scores


def search_threshold(
    scored_pairs,
    source1_ids,
    ground_truth_lookup: dict,
    thresholds=None,
    beta: float = None,
):
    """
    Parameters
    ----------
    scored_pairs : DataFrame with columns source1_entity_id,
        candidate_entity_id, score (predicted match probability)
    source1_ids : all Source 1 entity_ids in the validation split (so that
        singletons with zero candidates are still scored, not skipped)
    ground_truth_lookup : {source1_entity_id: comma-joined matched IDs str}
    thresholds : iterable of thresholds to try; defaults to a fine grid.
    beta : F-beta parameter; defaults to config.F_BETA.

    Returns
    -------
    (best_threshold, best_score, all_results) where all_results is a list of
    (threshold, macro_f_beta) tuples for inspection/plotting.
    """
    beta = beta if beta is not None else config.F_BETA
    thresholds = thresholds if thresholds is not None else np.linspace(0.05, 0.95, 19)

    results = []
    best_threshold, best_score = None, -1.0

    for t in thresholds:
        predictions = predictions_from_scores(scored_pairs, source1_ids, threshold=t)
        eval_result = macro_f_beta(predictions, ground_truth_lookup, beta=beta)
        score = eval_result["macro_f_beta"]
        results.append((t, score))
        if score > best_score:
            best_threshold, best_score = t, score

    return best_threshold, best_score, results
