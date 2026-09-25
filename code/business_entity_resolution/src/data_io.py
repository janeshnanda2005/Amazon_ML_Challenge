"""
Loading helpers for the challenge's TSV files.

Everything is read with `sep="\t"` and `dtype=str` - IDs and postal codes
must never be silently coerced to numbers, and business_address /
matched_entity_ids columns contain commas, which is exactly why the
challenge uses tabs as the delimiter in the first place.
"""

import pandas as pd


def read_source(path) -> pd.DataFrame:
    """Load a *_source{1,2,3}.tsv file: entity_id, business_name,
    business_address, country."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    expected = {"entity_id", "business_name", "business_address", "country"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    return df


def read_ground_truth(path) -> pd.DataFrame:
    """Load train_ground_truth.tsv: source1_entity_id, matched_entity_ids."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    expected = {"source1_entity_id", "matched_entity_ids"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    return df


def stack_candidate_pool(source2: pd.DataFrame, source3: pd.DataFrame) -> pd.DataFrame:
    """Concatenate Source 2 + Source 3 into one candidate pool for blocking.
    entity_id prefixes (S2-/S3-) already disambiguate the origin source."""
    return pd.concat([source2, source3], ignore_index=True)
