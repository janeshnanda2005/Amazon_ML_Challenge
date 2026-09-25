"""
Central configuration for the Business Entity Resolution pipeline.

Every other module imports paths and tunables from here so there is exactly
one place to change when you move from a laptop sample to the full ~1.7M
row test set.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to the `business_entity_resolution/` project root)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = ROOT / "dataset" / "train"
TEST_DIR = ROOT / "dataset" / "test"
OUTPUT_DIR = ROOT / "output"
MODEL_DIR = ROOT / "models"

TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"
MODEL_PATH = MODEL_DIR / "matcher.joblib"

# ---------------------------------------------------------------------------
# Blocking
# ---------------------------------------------------------------------------
# Minimum shared-token count for the token-overlap blocking strategy.
MIN_SHARED_NAME_TOKENS = 1

# Cap on candidates generated per Source-1 entity, purely as a safety valve
# against pathological blocking keys (e.g. a very common token). Tune this
# up if you see blocking recall suffering on the training split.
MAX_CANDIDATES_PER_ENTITY = 200

# Stopword-like business tokens that are too common to block on alone
# (legal suffixes, generic words). Expand this after inspecting your data.
GENERIC_NAME_TOKENS = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "llc", "llp", "pvt", "private", "plc",
    "group", "holdings", "enterprises", "services", "solutions",
    "the", "and", "of", "&",
}

# ---------------------------------------------------------------------------
# Feature engineering / modelling
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# Fraction of Source-1 training entities held out for threshold tuning /
# validation. Split by entity (never by pair) to avoid leakage.
VALIDATION_FRACTION = 0.2

# F-beta used by the competition (precision-heavy: beta < 1)
F_BETA = 0.5
