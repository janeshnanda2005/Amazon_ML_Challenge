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
    if "norm_name" not in out.columns:
        out["norm_name"] = out["business_name"].map(normalize_name)
    if "norm_addr" not in out.columns:
        out["norm_addr"] = out["business_address"].map(normalize_address)
    if "name_tokens" not in out.columns:
        out["name_tokens"] = out["norm_name"].map(tokenize)
    if "addr_tokens" not in out.columns:
        out["addr_tokens"] = out["norm_addr"].map(tokenize)
    if "postal" not in out.columns:
        if "postal_code" in out.columns:
            out["postal"] = out["postal_code"]
        else:
            out["postal"] = out["business_address"].map(extract_postal_token)
    if "name_soundex" not in out.columns:
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
    if len(candidate_pairs) == 0:
        empty_feats = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"] + FEATURE_COLUMNS)
        return empty_feats, tfidf_vectorizer

    needed_s1 = set(candidate_pairs["source1_entity_id"].unique())
    needed_pool = set(candidate_pairs["candidate_entity_id"].unique())

    s1_sub = source1[source1["entity_id"].isin(needed_s1)]
    if hasattr(candidate_pool, "fetch_candidate_pool_records"):
        pool_sub = candidate_pool.fetch_candidate_pool_records(list(needed_pool))
    else:
        pool_sub = candidate_pool[candidate_pool["entity_id"].isin(needed_pool)]

    s1_prep = _prep_records(s1_sub).set_index("entity_id")
    pool_prep = _prep_records(pool_sub).set_index("entity_id")

    s1_ids = candidate_pairs["source1_entity_id"].tolist()
    cand_ids = candidate_pairs["candidate_entity_id"].tolist()

    # Fast flat dict lookups
    s1_norm_name = s1_prep["norm_name"].to_dict()
    pool_norm_name = pool_prep["norm_name"].to_dict()
    s1_norm_addr = s1_prep["norm_addr"].to_dict()
    pool_norm_addr = pool_prep["norm_addr"].to_dict()
    s1_tokens = s1_prep["name_tokens"].to_dict()
    pool_tokens = pool_prep["name_tokens"].to_dict()
    s1_addr_tokens = s1_prep["addr_tokens"].to_dict()
    pool_addr_tokens = pool_prep["addr_tokens"].to_dict()
    s1_soundex = s1_prep["name_soundex"].to_dict()
    pool_soundex = pool_prep["name_soundex"].to_dict()
    s1_postal = s1_prep["postal"].to_dict()
    pool_postal = pool_prep["postal"].to_dict()
    s1_country = s1_prep["country"].to_dict()
    pool_country = pool_prep["country"].to_dict()

    name_a = [s1_norm_name.get(s, "") for s in s1_ids]
    name_b = [pool_norm_name.get(c, "") for c in cand_ids]
    addr_a = [s1_norm_addr.get(s, "") for s in s1_ids]
    addr_b = [pool_norm_addr.get(c, "") for c in cand_ids]
    toks_a = [s1_tokens.get(s, []) for s in s1_ids]
    toks_b = [pool_tokens.get(c, []) for c in cand_ids]
    atok_a = [s1_addr_tokens.get(s, []) for s in s1_ids]
    atok_b = [pool_addr_tokens.get(c, []) for c in cand_ids]
    sdx_a = [s1_soundex.get(s, "") for s in s1_ids]
    sdx_b = [pool_soundex.get(c, "") for c in cand_ids]
    post_a = [s1_postal.get(s, "") for s in s1_ids]
    post_b = [pool_postal.get(c, "") for c in cand_ids]
    cntry_a = [s1_country.get(s, "") for s in s1_ids]
    cntry_b = [pool_country.get(c, "") for c in cand_ids]

    feats = pd.DataFrame(index=candidate_pairs.index)
    feats["source1_entity_id"] = s1_ids
    feats["candidate_entity_id"] = cand_ids

    feats["name_levenshtein"] = [
        levenshtein_ratio(a, b) for a, b in zip(name_a, name_b)
    ]
    feats["name_jaro_winkler"] = [
        jaro_winkler(a, b) for a, b in zip(name_a, name_b)
    ]
    feats["name_token_jaccard"] = [
        token_jaccard(a, b) for a, b in zip(toks_a, toks_b)
    ]
    feats["name_token_sort"] = [
        token_sort_ratio(a, b) for a, b in zip(toks_a, toks_b)
    ]
    feats["name_soundex_match"] = [
        int(a == b and a != "") for a, b in zip(sdx_a, sdx_b)
    ]
    feats["name_length_diff"] = [
        abs(len(a) - len(b)) for a, b in zip(name_a, name_b)
    ]
    feats["num_common_name_tokens"] = [
        len(set(a) & set(b)) for a, b in zip(toks_a, toks_b)
    ]

    feats["addr_levenshtein"] = [
        levenshtein_ratio(a, b) for a, b in zip(addr_a, addr_b)
    ]
    feats["addr_jaro_winkler"] = [
        jaro_winkler(a, b) for a, b in zip(addr_a, addr_b)
    ]
    feats["addr_token_jaccard"] = [
        token_jaccard(a, b) for a, b in zip(atok_a, atok_b)
    ]
    feats["addr_length_diff"] = [
        abs(len(a) - len(b)) for a, b in zip(addr_a, addr_b)
    ]

    feats["postal_present_both"] = [
        int(bool(a) and bool(b)) for a, b in zip(post_a, post_b)
    ]
    feats["postal_match"] = [
        int(bool(a) and a == b) for a, b in zip(post_a, post_b)
    ]

    feats["country_match"] = [
        int(str(a).strip().lower() == str(b).strip().lower())
        for a, b in zip(cntry_a, cntry_b)
    ]

    if tfidf_vectorizer is None:
        corpus = pd.concat([s1_prep["norm_name"], pool_prep["norm_name"]]).drop_duplicates()
        if len(corpus) > 200_000:
            corpus = corpus.sample(200_000, random_state=42)
        tfidf_vectorizer = build_char_tfidf_vectorizer(corpus.tolist())

    feats["name_tfidf_cosine"] = tfidf_cosine_pairs(
        tfidf_vectorizer, name_a, name_b
    )
    feats["addr_tfidf_cosine"] = tfidf_cosine_pairs(
        tfidf_vectorizer, addr_a, addr_b
    )

    return feats, tfidf_vectorizer
