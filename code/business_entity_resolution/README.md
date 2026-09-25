# Business Entity Resolution — ML Challenge 2026

End-to-end pipeline: **blocking → pairwise feature engineering → gradient-boosted
classifier → threshold search on macro F₀.₅ → submission files**.

## Setup

```bash
cd business_entity_resolution
pip install -r requirements.txt
```

Place the challenge data exactly as provided:

```
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

## Run

```bash
# 1. Train: blocking recall report, classifier, threshold search, saved model
python -m src.pipeline_train

# 2. Predict: candidate generation + scoring + submission files
python -m src.pipeline_predict

# 3. Validate before uploading
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

`pipeline_train.py` writes `models/matcher.joblib` (classifier + fitted
TF-IDF vectorizer) and `models/threshold.json` (chosen decision threshold +
its validation macro F₀.₅). `pipeline_predict.py` reads both and writes
`output/matching_results.tsv` + `output/candidate_pairs.tsv`.

## Pipeline stages (see module docstrings for full detail)

| Stage | Module | What it does |
|---|---|---|
| Normalization | `src/normalize.py` | Lowercase, strip punctuation, expand abbreviations (Corp→Corporation, Rd→Road, etc.), extract postal tokens |
| Blocking | `src/blocking.py` | Inverted indexes over Source 2+3 (name token, phonetic/Soundex, postal code, address token); unioned per Source-1 query; `blocking_recall()` measures the recall ceiling |
| Labeling | `src/labeling.py` | Joins candidate pairs against ground truth; splits by **entity**, never by pair |
| Features | `src/features.py` | 16 pairwise similarity features (Levenshtein, Jaro-Winkler, token Jaccard/sort-ratio, TF-IDF cosine, Soundex match, postal match, country match, length diffs) |
| Model | `src/model.py` | Gradient-boosted trees (sklearn `HistGradientBoostingClassifier` by default; swap to LightGBM per the docstring) |
| Threshold | `src/threshold.py` | Grid-searches the decision cutoff directly against **macro F₀.₅ including singletons** — not pair-level precision/recall |
| Evaluation | `src/evaluate.py` | Exact reimplementation of the competition's scoring: per-entity F₀.₅, macro-averaged, singletons scored |
| Output | `src/output.py` | Writes both required TSVs in the exact validator-checked format |

## Known simplifications / what to change for the real ~1.7M-row test set

1. **String similarity**: `src/similarity.py` ships pure-Python Levenshtein/
   Jaro-Winkler so this runs with zero extra dependencies. Swap to
   `rapidfuzz` (see that module's docstring) for the real run — it's
   10-100x faster in C and the call signatures are kept parallel on purpose.
2. **Classifier**: `src/model.py` defaults to sklearn's
   `HistGradientBoostingClassifier` for portability. Swap to LightGBM (see
   that module's docstring) for faster training and typically better
   tabular performance at scale.
3. **Blocking candidate cap**: `config.MAX_CANDIDATES_PER_ENTITY` truncates
   pathological candidate sets (e.g. a very generic name token). If you see
   this triggering often in your blocking-recall report, tighten a blocking
   key (raise `GENERIC_NAME_TOKENS`/`GENERIC_ADDRESS_TOKENS` coverage)
   rather than just raising the cap.
4. This repo ships `make_synthetic_data.py`, used only to prove the
   pipeline runs end-to-end without the real dataset. **Delete it** (or
   just don't run it) once you're pointing the pipeline at real data.

## Fair-play note

No external lookups of any kind are used anywhere in this pipeline —
everything (normalization, blocking, features, model) is derived only from
`train_source1/2/3.tsv` and `train_ground_truth.tsv`, per the challenge
rules.
