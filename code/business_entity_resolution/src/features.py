"""
Pairwise feature engineering.

Takes a long-format table of candidate pairs (source1_entity_id,
candidate_entity_id) plus the source record tables, and produces a numeric
feature matrix - one row per pair - for the classifier.

Kept as plain pandas + the similarity primitives in `similarity.py`. At full
scale, swap `similarity.levenshtein_ratio` / `jaro_winkler` for rapidfuzz's
C implementations (see the docstring in similarity.py) - the column names
and feature semantics below don't need to change.
"""

import pandas as pd

from .normalize import (
    normalize_name,
    normalize_address,
    tokenize,
    extract_postal_token,
)
from .similarity import (
    levenshtein_ratio,
    jaro_winkler,
    token_jaccard,
    token_sort_ratio,
    soundex,
    build_char_tfidf_vectorizer,
    tfidf_cosine_pairs,
)

FEATURE_COLUMNS = [
    "name_levenshtein",
    "name_jaro_winkler",
    "name_token_jaccard",
    "name_token_sort",
    "name_tfidf_cosine",
    "name_soundex_match",
    "name_length_diff",
    "num_common_name_tokens",
    "addr_levenshtein",
    "addr_jaro_winkler",
    "addr_token_jaccard",
    "addr_tfidf_cosine",
    "addr_length_diff",
    "postal_present_both",
    "postal_match",
    "country_match",
]


def _prep_records(records: pd.DataFrame) -> pd.DataFrame:
    """Precompute normalized fields once per record (not per pair)."""
    out = records.copy()
    out["norm_name"] = out["business_name"].map(normalize_name)
    out["norm_addr"] = out["business_address"].map(normalize_address)
    out["name_tokens"] = out["norm_name"].map(tokenize)
    out["addr_tokens"] = out["norm_addr"].map(tokenize)
    out["postal"] = out["business_address"].map(extract_postal_token)
    out["name_soundex"] = out["name_tokens"].map(
        lambda toks: soundex(toks[0]) if toks else ""
    )
    return out


def build_feature_table(
    candidate_pairs: pd.DataFrame,
    source1: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    tfidf_vectorizer=None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    candidate_pairs : DataFrame[source1_entity_id, candidate_entity_id]
    source1 : DataFrame of Source 1 records (entity_id, business_name,
        business_address, country)
    candidate_pool : DataFrame of Source 2 + Source 3 records, same columns
    tfidf_vectorizer : optional pre-fit char-ngram TfidfVectorizer (fit over
        both name corpora); fit fresh on `source1` + `candidate_pool` if not
        given. Pass the *same* fitted vectorizer between train and inference
        so the feature space lines up.

    Returns
    -------
    DataFrame indexed identically to `candidate_pairs`, with the pair keys
    plus every column in FEATURE_COLUMNS.
    """
    s1 = _prep_records(source1).set_index("entity_id")
    pool = _prep_records(candidate_pool).set_index("entity_id")

    left = candidate_pairs["source1_entity_id"].map(s1.to_dict("index"))
    right = candidate_pairs["candidate_entity_id"].map(pool.to_dict("index"))

    feats = pd.DataFrame(index=candidate_pairs.index)
    feats["source1_entity_id"] = candidate_pairs["source1_entity_id"].values
    feats["candidate_entity_id"] = candidate_pairs["candidate_entity_id"].values

    name_a = left.map(lambda d: d["norm_name"])
    name_b = right.map(lambda d: d["norm_name"])
    addr_a = left.map(lambda d: d["norm_addr"])
    addr_b = right.map(lambda d: d["norm_addr"])

    feats["name_levenshtein"] = [
        levenshtein_ratio(a, b) for a, b in zip(name_a, name_b)
    ]
    feats["name_jaro_winkler"] = [
        jaro_winkler(a, b) for a, b in zip(name_a, name_b)
    ]
    feats["name_token_jaccard"] = [
        token_jaccard(a["name_tokens"], b["name_tokens"])
        for a, b in zip(left, right)
    ]
    feats["name_token_sort"] = [
        token_sort_ratio(a["name_tokens"], b["name_tokens"])
        for a, b in zip(left, right)
    ]
    feats["name_soundex_match"] = [
        int(a["name_soundex"] == b["name_soundex"] and a["name_soundex"] != "")
        for a, b in zip(left, right)
    ]
    feats["name_length_diff"] = [
        abs(len(a) - len(b)) for a, b in zip(name_a, name_b)
    ]
    feats["num_common_name_tokens"] = [
        len(set(a["name_tokens"]) & set(b["name_tokens"]))
        for a, b in zip(left, right)
    ]

    feats["addr_levenshtein"] = [
        levenshtein_ratio(a, b) for a, b in zip(addr_a, addr_b)
    ]
    feats["addr_jaro_winkler"] = [
        jaro_winkler(a, b) for a, b in zip(addr_a, addr_b)
    ]
    feats["addr_token_jaccard"] = [
        token_jaccard(a["addr_tokens"], b["addr_tokens"])
        for a, b in zip(left, right)
    ]
    feats["addr_length_diff"] = [
        abs(len(a) - len(b)) for a, b in zip(addr_a, addr_b)
    ]

    postal_a = left.map(lambda d: d["postal"])
    postal_b = right.map(lambda d: d["postal"])
    feats["postal_present_both"] = [
        int(bool(a) and bool(b)) for a, b in zip(postal_a, postal_b)
    ]
    feats["postal_match"] = [
        int(bool(a) and a == b) for a, b in zip(postal_a, postal_b)
    ]

    country_a = left.map(lambda d: d.get("country", ""))
    country_b = right.map(lambda d: d.get("country", ""))
    feats["country_match"] = [
        int(str(a).strip().lower() == str(b).strip().lower())
        for a, b in zip(country_a, country_b)
    ]

    # TF-IDF cosine similarity is vectorized (not per-pair Python), so it's
    # computed last, over the whole batch at once.
    if tfidf_vectorizer is None:
        corpus = pd.concat([s1["norm_name"], pool["norm_name"]]).unique().tolist()
        tfidf_vectorizer = build_char_tfidf_vectorizer(corpus)
    feats["name_tfidf_cosine"] = tfidf_cosine_pairs(
        tfidf_vectorizer, name_a.tolist(), name_b.tolist()
    )
    feats["addr_tfidf_cosine"] = tfidf_cosine_pairs(
        tfidf_vectorizer, addr_a.tolist(), addr_b.tolist()
    )

    return feats, tfidf_vectorizer
