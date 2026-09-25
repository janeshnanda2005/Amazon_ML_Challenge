"""
String similarity primitives used for both blocking (phonetic keys) and
pairwise feature engineering.

Production note
----------------
This module ships pure-Python / numpy implementations so the pipeline runs
with zero extra dependencies. On the real ~1.7M-row test set these are too
slow for the *feature engineering* step (millions of pairwise comparisons).
Once you have network access, install `rapidfuzz` (MIT-licensed, drop-in
replacement) and swap:

    from rapidfuzz.distance import Levenshtein, JaroWinkler
    from rapidfuzz.fuzz import token_sort_ratio, token_set_ratio

    levenshtein_ratio(a, b)   -> Levenshtein.normalized_similarity(a, b)
    jaro_winkler(a, b)        -> JaroWinkler.similarity(a, b)

rapidfuzz is 10-100x faster in C and the call signatures below are kept
close to it on purpose to make that swap mechanical.
"""

from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# Levenshtein distance / ratio
# ---------------------------------------------------------------------------
@lru_cache(maxsize=200_000)
def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr_row = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr_row[j] = min(
                prev_row[j] + 1,       # deletion
                curr_row[j - 1] + 1,   # insertion
                prev_row[j - 1] + cost,  # substitution
            )
        prev_row = curr_row
    return prev_row[-1]


def levenshtein_ratio(a: str, b: str) -> float:
    """Normalized similarity in [0, 1]; 1.0 means identical strings."""
    if not a and not b:
        return 1.0
    dist = levenshtein_distance(a, b)
    return 1.0 - dist / max(len(a), len(b))


# ---------------------------------------------------------------------------
# Jaro-Winkler similarity
# ---------------------------------------------------------------------------
def jaro_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0

    match_distance = max(la, lb) // 2 - 1
    match_distance = max(match_distance, 0)

    a_matches = [False] * la
    b_matches = [False] * lb

    matches = 0
    transpositions = 0

    for i in range(la):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, lb)
        for j in range(start, end):
            if b_matches[j] or a[i] != b[j]:
                continue
            a_matches[i] = True
            b_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(la):
        if not a_matches[i]:
            continue
        while not b_matches[k]:
            k += 1
        if a[i] != b[k]:
            transpositions += 1
        k += 1
    transpositions //= 2

    return (
        matches / la + matches / lb + (matches - transpositions) / matches
    ) / 3.0


def jaro_winkler(a: str, b: str, prefix_weight: float = 0.1) -> float:
    if a == b:
        return 1.0
    jaro = jaro_similarity(a, b)
    prefix_len = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        prefix_len += 1
        if prefix_len == 4:
            break
    return jaro + prefix_len * prefix_weight * (1 - jaro)


# ---------------------------------------------------------------------------
# Token-set similarity
# ---------------------------------------------------------------------------
def token_jaccard(tokens_a, tokens_b) -> float:
    set_a, set_b = set(tokens_a), set(tokens_b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def token_sort_ratio(tokens_a, tokens_b) -> float:
    """Levenshtein ratio after sorting tokens - robust to word-order swaps."""
    a_sorted = " ".join(sorted(tokens_a))
    b_sorted = " ".join(sorted(tokens_b))
    return levenshtein_ratio(a_sorted, b_sorted)


# ---------------------------------------------------------------------------
# Soundex (simple phonetic key, used for blocking)
# ---------------------------------------------------------------------------
_SOUNDEX_CODES = {
    **{c: "1" for c in "bfpv"},
    **{c: "2" for c in "cgjkqsxz"},
    **{c: "3" for c in "dt"},
    "l": "4",
    **{c: "5" for c in "mn"},
    "r": "6",
}


def soundex(word: str) -> str:
    """Classic Soundex phonetic key - used only as a blocking key, never a
    feature (it's too coarse for scoring)."""
    word = "".join(c for c in word.lower() if c.isalpha())
    if not word:
        return "0000"
    first_letter = word[0]
    codes = [_SOUNDEX_CODES.get(c, "") for c in word[1:]]

    deduped = []
    prev = _SOUNDEX_CODES.get(word[0], "")
    for c in codes:
        if c != prev:
            deduped.append(c)
        prev = c
    digits = "".join(deduped).replace("", "")[:3]
    digits = digits.ljust(3, "0")
    return (first_letter + digits).upper()


# ---------------------------------------------------------------------------
# TF-IDF cosine similarity (batch, vectorized - use for scoring many pairs)
# ---------------------------------------------------------------------------
def build_char_tfidf_vectorizer(corpus, ngram_range=(2, 4)):
    """Fit a character n-gram TF-IDF vectorizer over a corpus of normalized
    strings (typically all names or all addresses seen in train+test)."""
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=ngram_range, min_df=1
    )
    vectorizer.fit(corpus)
    return vectorizer


def tfidf_cosine_pairs(vectorizer, texts_a, texts_b) -> np.ndarray:
    """Row-wise cosine similarity between texts_a[i] and texts_b[i] for a
    batch of pairs, using a pre-fit vectorizer. Vectorized - safe at scale."""
    mat_a = vectorizer.transform(texts_a)
    mat_b = vectorizer.transform(texts_b)
    # Rows are L2-normalized by TfidfVectorizer's default norm="l2", so the
    # dot product of matching rows IS the cosine similarity.
    dots = np.asarray(mat_a.multiply(mat_b).sum(axis=1)).ravel()
    return dots
