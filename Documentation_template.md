# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Codeholics
**Team Members:** Janeshnanda K S, Sreesanth S, Jeffery, Seetharaman L

---

## 1. Executive Summary

We solve business entity resolution as pairwise binary classification over
a blocked candidate set: multiple independent inverted-index blocking
strategies (name token, phonetic Soundex, postal code, address token) are
unioned to protect recall, sixteen string/token/postal/country similarity
features are computed per candidate pair, and a gradient-boosted tree
classifier scores each pair. The decision threshold is tuned directly
against the competition's own metric — macro-averaged F₀.₅ per Source-1
entity, including singletons — rather than against pooled precision/recall.

---

## 2. Methodology

### 2.1 Problem Analysis

Key noise patterns anticipated from the problem statement and handled
explicitly:

- **Legal-suffix / abbreviation variation** (Corp/Corporation, Pvt/Private,
  Ltd/Limited, & vs "and") — normalized via an explicit abbreviation-
  expansion map (`src/normalize.py`) before any similarity scoring or
  blocking key is computed.
- **Address abbreviation variation** (Rd/Road, St/Street) — same
  expansion-map approach, address-specific.
- **Missing address components** (no PIN/postal code, no state) — postal
  match is treated as a *feature*, not a filter: `postal_present_both` and
  `postal_match` are separate features, so a missing postal code lowers
  confidence rather than eliminating the pair from consideration.
- **Open-set `country`** — the test set introduces `France`, unseen in
  training. `country` is never one-hot encoded or filtered on; it is used
  only as a same/different binary feature (`country_match`), which
  generalizes to unseen values by construction.
- **Word-order transpositions** — handled by the token-sort-ratio feature
  (sorts tokens before computing Levenshtein similarity).
- **Transliteration / typo variants** — handled by the phonetic (Soundex)
  blocking key and by character n-gram TF-IDF cosine similarity, which
  degrades more gracefully than exact/edit-distance methods under heavy
  spelling drift.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (not end-to-end deep learning).

**Core Innovation:** Four independent blocking strategies are unioned
rather than intersected or run as a single pass, specifically so that a
single key type's blind spot (e.g., a missing postal code, or a name
typo that breaks phonetic matching) does not cap recall — as long as
*any one* strategy recovers the pair, it survives to the classifier. This
is checked quantitatively via `blocking.blocking_recall()` against
`train_ground_truth.tsv` before any modeling work is trusted.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  1. Normalized, legal-suffix-stripped name tokens (inverted index)
  2. Soundex phonetic key of the first core name token
  3. Extracted postal/PIN code token from the address (last standalone
     4–6 digit run)
  4. Normalized address tokens, excluding generic words (street, road,
     near, floor, etc.) and pure digits

- **Candidate pairs generated:** [fill in from your `blocking_recall()`
  log on the real training set — `candidate_pairs_total`]

- **How true matches were not lost:** Recall is measured directly, not
  assumed. `blocking.blocking_recall()` compares the generated candidate
  set against every matched ID in `train_ground_truth.tsv` and reports
  the fraction recovered. `pipeline_train.py` prints this automatically
  and emits a warning if recall falls below 0.95 — a threshold you should
  keep investigating against (widen a blocking key, e.g. lower the
  generic-token threshold or add a secondary phonetic key) until the
  ceiling this imposes on your final F₀.₅ is acceptable.

  On our synthetic smoke-test dataset (used only to validate the pipeline
  runs correctly, not representative of real performance): 253/271 true
  matches recovered → **93.4% blocking recall**, 60,000 total candidate
  pairs generated from 300 Source-1 entities. *Replace this paragraph with
  your real training-set numbers.*

---

## 4. Matching Model

**Features used** (16 total, see `src/features.py::FEATURE_COLUMNS`):

- Name features: Levenshtein ratio, Jaro-Winkler similarity, token
  Jaccard, token-sort ratio, character n-gram TF-IDF cosine, Soundex
  match (binary), length difference, count of shared tokens
- Address features: Levenshtein ratio, Jaro-Winkler similarity, token
  Jaccard, character n-gram TF-IDF cosine, length difference
- Other: postal-code match (binary, gated on both records having a
  postal code present), country match (binary, open-set safe)

**Model type:** Gradient-boosted decision trees. Reference implementation
uses scikit-learn's `HistGradientBoostingClassifier` (portable, zero extra
dependencies); production recommendation is LightGBM (MIT-licensed, faster
at scale) — see `src/model.py` for the exact drop-in swap. Both comply
with the challenge's model constraint (MIT/Apache-2.0 license, well under
8B parameters — a tree ensemble has no "parameters" in the neural-network
sense and is not a language model).

**Threshold selection method:** Grid search over candidate thresholds,
each evaluated with the competition's exact scoring function
(`src/evaluate.py::macro_f_beta`) — per-Source-1-entity F₀.₅, macro-
averaged, **singletons included** (correct empty prediction = 1.0, any
false merge on a true singleton = 0.0). This is deliberately not a
pair-level precision/recall threshold search, since that would
over-weight entities with many candidates and under-weight singletons,
which this competition's metric penalizes heavily.

---

## 5. Results & Error Analysis

- **F₀.₅ Score (macro):** [fill in your best validation score from
  `models/threshold.json` after running `pipeline_train.py` on the real
  training data]
- **Common false positives (wrong merges):** [fill in after inspecting
  `evaluate.worst_entities()` output on your real validation split —
  typically near-duplicate business names at different real addresses,
  or generic name tokens like "General Store" colliding across unrelated
  businesses]
- **Common false negatives (missed matches):** [fill in — typically
  heavy transliteration drift beyond what Soundex captures, or addresses
  missing both postal code and recognizable street tokens]

---

## 6. Conclusion

[Summarize your approach, key achievements, and lessons learned in 2–3
sentences once run on the real dataset.]

---

## Appendix

### A. Code Artefacts

Complete, runnable code ships under `code/business_entity_resolution/src/`:

- `config.py` — paths and tunable constants
- `data_io.py` — TSV loading (tab-separated, dtype-safe)
- `normalize.py` — name/address cleaning and abbreviation expansion
- `similarity.py` — Levenshtein, Jaro-Winkler, Soundex, Jaccard, TF-IDF
  cosine primitives
- `blocking.py` — inverted-index candidate generation + recall diagnostic
- `labeling.py` — ground-truth join + entity-level train/val split
- `features.py` — pairwise feature table construction
- `model.py` — classifier training/scoring/persistence
- `threshold.py` — macro-F₀.₅-optimal threshold search
- `evaluate.py` — exact competition metric reimplementation
- `output.py` — submission TSV writers
- `pipeline_train.py` — entry point: `python -m src.pipeline_train`
- `pipeline_predict.py` — entry point: `python -m src.pipeline_predict`

See `code/business_entity_resolution/README.md` for exact reproduction
steps from raw data to `output/matching_results.tsv` and
`output/candidate_pairs.tsv`.

### B. Additional Results

[Add charts/tables from your real run here — e.g. threshold-vs-macro-F₀.₅
curve from `pipeline_train.py`'s printed grid, feature importances from
the trained model.]

---

**Note:** Teams can modify sections according to their approach while
maintaining clarity and technical depth.
