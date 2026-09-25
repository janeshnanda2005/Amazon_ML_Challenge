"""
Turn scored candidate pairs into:
  - a {source1_entity_id: predicted matched_entity_ids} dict (used both for
    offline evaluation and to write matching_results.tsv), and
  - the matching_results.tsv / candidate_pairs.tsv files themselves, in the
    exact format the challenge's validator expects.
"""

from collections import defaultdict

import pandas as pd


def predictions_from_scores(scored_pairs: pd.DataFrame, source1_ids, threshold: float) -> dict:
    """
    Apply a decision threshold and collapse to one comma-joined string of
    matched IDs per Source 1 entity. Every ID in `source1_ids` gets an
    entry (possibly empty) so singletons and no-candidate entities are
    represented - required by the "every Source 1 entity must appear"
    submission rule.
    """
    matches = defaultdict(list)
    above = scored_pairs[scored_pairs["score"] >= threshold]
    for s1_id, cand_id in zip(above["source1_entity_id"], above["candidate_entity_id"]):
        matches[s1_id].append(cand_id)

    return {
        s1_id: ",".join(dict.fromkeys(matches.get(s1_id, [])))  # de-dup, keep order
        for s1_id in source1_ids
    }


def candidates_from_pairs(candidate_pairs: pd.DataFrame, source1_ids) -> dict:
    """Same collapsing logic as predictions_from_scores, but for the raw
    (unfiltered) candidate set that fed the model - i.e. candidate_pairs.tsv."""
    grouped = defaultdict(list)
    for s1_id, cand_id in zip(
        candidate_pairs["source1_entity_id"], candidate_pairs["candidate_entity_id"]
    ):
        grouped[s1_id].append(cand_id)
    return {
        s1_id: ",".join(dict.fromkeys(grouped.get(s1_id, [])))
        for s1_id in source1_ids
    }


def write_results_tsv(id_to_matches: dict, path, id_col: str, match_col: str):
    """
    Write a {source1_entity_id: comma-joined-ids} dict as the challenge's
    two-column TSV format (matching_results.tsv / candidate_pairs.tsv share
    this exact shape).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(id_to_matches.items())  # deterministic output ordering
    df = pd.DataFrame(rows, columns=[id_col, match_col])
    df.to_csv(path, sep="\t", index=False)


def append_results_tsv(id_to_matches: dict, path, id_col: str, match_col: str, write_header: bool = False):
    """
    Append a batch of {source1_entity_id: comma-joined-ids} to TSV.
    Allows streaming test predictions in chunks without holding all results in RAM.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(id_to_matches.items())
    df = pd.DataFrame(rows, columns=[id_col, match_col])
    mode = "w" if write_header else "a"
    df.to_csv(path, sep="\t", index=False, mode=mode, header=write_header)


def write_matching_results(predictions: dict, path):
    write_results_tsv(predictions, path, "source1_entity_id", "matched_entity_ids")


def write_candidate_pairs(candidates: dict, path):
    write_results_tsv(candidates, path, "source1_entity_id", "candidate_entity_ids")

