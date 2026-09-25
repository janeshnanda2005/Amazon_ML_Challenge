"""
Macro-averaged F_beta scoring, matching the competition's evaluation
exactly: F_0.5 is computed per Source 1 entity (singletons included, scoring
1.0 for a correctly-predicted empty list and 0.0 for any false merge on a
true singleton), then averaged across all Source 1 entities.

This module is used both for offline validation-split scoring during
threshold search, and can be pointed at any (predictions, ground truth)
pair to sanity-check a submission before uploading it.
"""

from collections import defaultdict


def _f_beta(precision: float, recall: float, beta: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_sq = beta ** 2
    denom = beta_sq * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denom


def _parse_id_list(raw) -> set:
    raw = str(raw) if raw is not None else ""
    if not raw or raw.lower() == "nan":
        return set()
    return {x.strip() for x in raw.split(",") if x.strip()}


def macro_f_beta(
    predictions: dict, ground_truth: dict, beta: float = 0.5
) -> dict:
    """
    Parameters
    ----------
    predictions : {source1_entity_id: comma-joined matched_entity_ids str}
    ground_truth : {source1_entity_id: comma-joined matched_entity_ids str}
        Must contain every Source 1 entity that predictions should cover.
    beta : float, the F-beta parameter (0.5 for this competition).

    Returns
    -------
    dict with overall `macro_f_beta` plus per-entity scores for error
    analysis (e.g. to inspect the worst-scoring entities).
    """
    per_entity_scores = {}

    for s1_id, true_raw in ground_truth.items():
        true_set = _parse_id_list(true_raw)
        pred_set = _parse_id_list(predictions.get(s1_id, ""))

        if not true_set and not pred_set:
            # Correctly predicted singleton - full credit.
            per_entity_scores[s1_id] = 1.0
            continue
        if not true_set and pred_set:
            # False merge on a true singleton - zero credit.
            per_entity_scores[s1_id] = 0.0
            continue

        tp = len(true_set & pred_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(true_set) if true_set else 0.0
        per_entity_scores[s1_id] = _f_beta(precision, recall, beta)

    macro_score = (
        sum(per_entity_scores.values()) / len(per_entity_scores)
        if per_entity_scores
        else 0.0
    )

    return {
        "macro_f_beta": macro_score,
        "n_entities": len(per_entity_scores),
        "per_entity_scores": per_entity_scores,
    }


def worst_entities(eval_result: dict, n: int = 20):
    """Return the n lowest-scoring Source 1 entities, for error analysis."""
    scores = eval_result["per_entity_scores"]
    return sorted(scores.items(), key=lambda kv: kv[1])[:n]
