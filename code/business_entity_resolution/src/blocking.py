"""
Candidate generation (blocking).

Strategy: build several cheap inverted indexes over the union of Source 2 +
Source 3 records, then union the candidates each index produces for a given
Source 1 record. Using several independent, low-precision-but-high-recall
keys and unioning them is deliberate - it means one key type's blind spot
(e.g. a missing PIN code) doesn't cap recall, as long as another key type
(e.g. a shared name token) catches the pair.

Indexes built:
  1. name_token   - normalized, generic-suffix-stripped name tokens
  2. name_soundex - phonetic key of the first core name token (typos,
                    transliteration variants, minor misspellings)
  3. postal       - extracted postal/PIN code token from the address
  4. addr_token   - normalized address tokens (street/area names), with
                    very common tokens (numbers, generic words) skipped

All of this is vectorized with pandas groupby, not per-row Python loops -
required to stay tractable at ~1.7M-row scale.
"""

from collections import defaultdict

import pandas as pd

from . import config
from .normalize import (
    core_name_tokens,
    normalize_address,
    tokenize,
    extract_postal_token,
)
from .similarity import soundex

# Address tokens too generic to block on by themselves.
GENERIC_ADDRESS_TOKENS = {
    "street", "road", "avenue", "lane", "drive", "boulevard", "highway",
    "near", "number", "floor", "building", "apartment", "the", "and", "of",
}


class BlockIndex:
    """Inverted indexes over a pool of candidate records (Source 2 + 3)."""

    def __init__(self, records: pd.DataFrame):
        """
        Parameters
        ----------
        records : DataFrame with columns entity_id, business_name,
            business_address, country (Source 2 and Source 3 rows
            concatenated together - entity_id prefix disambiguates source).
        """
        self.name_token_index = defaultdict(set)
        self.soundex_index = defaultdict(set)
        self.postal_index = defaultdict(set)
        self.addr_token_index = defaultdict(set)

        self._build(records)

    def _build(self, records: pd.DataFrame):
        for row in records.itertuples(index=False):
            entity_id = row.entity_id
            name_tokens = core_name_tokens(
                row.business_name, config.GENERIC_NAME_TOKENS
            )
            for tok in name_tokens:
                self.name_token_index[tok].add(entity_id)

            if name_tokens:
                key = soundex(name_tokens[0])
                self.soundex_index[key].add(entity_id)

            postal = extract_postal_token(row.business_address)
            if postal:
                self.postal_index[postal].add(entity_id)

            addr_tokens = tokenize(normalize_address(row.business_address))
            for tok in addr_tokens:
                if tok in GENERIC_ADDRESS_TOKENS or tok.isdigit():
                    continue
                self.addr_token_index[tok].add(entity_id)

    def candidates_for(self, business_name: str, business_address: str) -> set:
        """Union of candidate entity_ids across all blocking strategies."""
        candidates = set()

        name_tokens = core_name_tokens(business_name, config.GENERIC_NAME_TOKENS)
        for tok in name_tokens:
            candidates |= self.name_token_index.get(tok, set())

        if name_tokens:
            key = soundex(name_tokens[0])
            candidates |= self.soundex_index.get(key, set())

        postal = extract_postal_token(business_address)
        if postal:
            candidates |= self.postal_index.get(postal, set())

        addr_tokens = tokenize(normalize_address(business_address))
        for tok in addr_tokens:
            if tok in GENERIC_ADDRESS_TOKENS or tok.isdigit():
                continue
            candidates |= self.addr_token_index.get(tok, set())

        return candidates


def generate_candidate_pairs(
    source1: pd.DataFrame, index: BlockIndex, max_candidates: int = None
) -> pd.DataFrame:
    """
    For every Source 1 record, look up candidates from `index`.

    Returns a long-format DataFrame with one row per (source1_entity_id,
    candidate_entity_id) pair - the natural shape for the feature-engineering
    step, and easy to collapse into the TSV's comma-joined-list format later.
    """
    max_candidates = max_candidates or config.MAX_CANDIDATES_PER_ENTITY

    rows = []
    for row in source1.itertuples(index=False):
        candidates = index.candidates_for(row.business_name, row.business_address)
        if len(candidates) > max_candidates:
            # Arbitrary but deterministic truncation; if you hit this often,
            # raise max_candidates or tighten a blocking key instead of
            # silently dropping true matches.
            candidates = set(sorted(candidates)[:max_candidates])
        for cand_id in candidates:
            rows.append((row.entity_id, cand_id))

    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"])


def blocking_recall(
    candidate_pairs: pd.DataFrame, ground_truth: pd.DataFrame
) -> dict:
    """
    Diagnostic: what fraction of true matches survive blocking?

    This is the single most important number to check before investing in
    the classifier - it is the hard ceiling on your final recall.
    """
    candidate_set = set(
        zip(candidate_pairs["source1_entity_id"], candidate_pairs["candidate_entity_id"])
    )

    total_true_matches = 0
    recovered_true_matches = 0

    for row in ground_truth.itertuples(index=False):
        matched_ids = [
            m.strip() for m in str(row.matched_entity_ids).split(",") if m.strip()
        ]
        total_true_matches += len(matched_ids)
        for mid in matched_ids:
            if (row.source1_entity_id, mid) in candidate_set:
                recovered_true_matches += 1

    recall = (
        recovered_true_matches / total_true_matches if total_true_matches else 1.0
    )
    return {
        "true_matches_total": total_true_matches,
        "true_matches_recovered": recovered_true_matches,
        "blocking_recall": recall,
        "candidate_pairs_total": len(candidate_set),
    }
