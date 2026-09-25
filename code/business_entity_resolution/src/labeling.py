"""
Turn `train_ground_truth.tsv` into a binary label per candidate pair.

A pair is a positive example if it appears in the ground truth's matched-ID
list for that Source 1 entity; every other *generated candidate* pair is a
negative example. Pairs that blocking never generated are simply absent
from the training set - this is exactly why blocking recall matters so
much, see `blocking.blocking_recall`.
"""

import pandas as pd


def ground_truth_pairs(ground_truth: pd.DataFrame) -> set:
    """Flatten train_ground_truth.tsv into a set of (s1_id, matched_id)."""
    pairs = set()
    for row in ground_truth.itertuples(index=False):
        matched = str(row.matched_entity_ids)
        if not matched or matched.lower() == "nan":
            continue
        for mid in matched.split(","):
            mid = mid.strip()
            if mid:
                pairs.add((row.source1_entity_id, mid))
    return pairs


def label_candidate_pairs(
    candidate_pairs: pd.DataFrame, ground_truth: pd.DataFrame
) -> pd.DataFrame:
    """Add a `label` column (1 = true match, 0 = non-match) to candidate_pairs."""
    true_pairs = ground_truth_pairs(ground_truth)
    labeled = candidate_pairs.copy()
    labeled["label"] = [
        int((s1, cand) in true_pairs)
        for s1, cand in zip(
            labeled["source1_entity_id"], labeled["candidate_entity_id"]
        )
    ]
    return labeled


def entity_level_split(source1_ids, validation_fraction: float, random_state: int):
    """
    Split Source 1 entity IDs into train/validation sets.

    Splitting by *entity* (not by pair) is required: if the same Source 1
    entity's pairs appeared in both train and validation, the model would
    be validated on an entity it partially learned from, inflating the
    validation score.
    """
    import numpy as np

    rng = np.random.RandomState(random_state)
    ids = list(source1_ids)
    rng.shuffle(ids)
    cutoff = int(len(ids) * (1 - validation_fraction))
    return set(ids[:cutoff]), set(ids[cutoff:])
