"""
Central configuration for the Business Entity Resolution pipeline.

Configurable through environment variables with safe defaults for 8 GB RAM machines.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Base Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

# Robust REPO_ROOT finding: look upwards for repo indicators (.git, real, or utils)
REPO_ROOT = ROOT.parent
for p in [ROOT, ROOT.parent, ROOT.parent.parent]:
    if (p / ".git").exists() or (p / "real").exists() or (p / "utils").exists():
        REPO_ROOT = p
        break

# Raw dataset directory
REAL_DIR = Path(os.environ.get("AMAZON_ML_DATA_DIR", REPO_ROOT / "real"))
TOP_DATASET = REPO_ROOT / "dataset"

if (REAL_DIR / "train" / "train_source1.tsv").exists() and (REAL_DIR / "test" / "test_source1.tsv").exists():
    TRAIN_DIR = REAL_DIR / "train"
    TEST_DIR = REAL_DIR / "test"
elif (TOP_DATASET / "train" / "train_source1.tsv").exists() and (TOP_DATASET / "test" / "test_source1.tsv").exists():
    TRAIN_DIR = TOP_DATASET / "train"
    TEST_DIR = TOP_DATASET / "test"
else:
    TRAIN_DIR = ROOT / "dataset" / "train"
    TEST_DIR = ROOT / "dataset" / "test"

# Parquet storage
PARQUET_DIR = Path(os.environ.get("PARQUET_DIR", REPO_ROOT / "data" / "parquet"))
PARQUET_TRAIN_DIR = PARQUET_DIR / "train"
PARQUET_TEST_DIR = PARQUET_DIR / "test"

PARQUET_TRAIN_SOURCE1 = PARQUET_TRAIN_DIR / "source1.parquet"
PARQUET_TRAIN_POOL = PARQUET_TRAIN_DIR / "pool.parquet"
PARQUET_TRAIN_GROUND_TRUTH = PARQUET_TRAIN_DIR / "ground_truth.parquet"

PARQUET_TEST_SOURCE1 = PARQUET_TEST_DIR / "source1.parquet"
PARQUET_TEST_POOL = PARQUET_TEST_DIR / "pool.parquet"

# DuckDB Database path (can be in-memory ":memory:" or file on SSD)
DUCKDB_DATABASE = os.environ.get("DUCKDB_DATABASE", str(REPO_ROOT / "data" / "duckdb.db"))
DUCKDB_MEMORY_LIMIT = os.environ.get("DUCKDB_MEMORY_LIMIT", "4GB")

# Outputs and Models
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", REPO_ROOT / "output" if (REPO_ROOT / "output").exists() else (ROOT / "output")))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models"))

TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"
CHECKPOINT_PATH = OUTPUT_DIR / "predict_checkpoint.json"
MODEL_PATH = MODEL_DIR / "matcher.joblib"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"

# ---------------------------------------------------------------------------
# Streaming & Chunking Configuration
# ---------------------------------------------------------------------------
# Source-1 entity chunk size during inference and feature extraction
SOURCE1_CHUNK_SIZE = int(os.environ.get("SOURCE1_CHUNK_SIZE", "2000"))

# Batch size for internal vectorized processing
FEATURE_BATCH_SIZE = int(os.environ.get("FEATURE_BATCH_SIZE", "5000"))

# Sample size for fitting character TF-IDF vectorizer (reused for all chunks)
TFIDF_SAMPLE_SIZE = int(os.environ.get("TFIDF_SAMPLE_SIZE", "100000"))

# CPU threads allocated to LightGBM and DuckDB
LIGHTGBM_THREADS = int(os.environ.get("LIGHTGBM_THREADS", "4"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(LIGHTGBM_THREADS))

# Maximum candidate pairs permitted per Source 1 entity (safety cap for pathological keys)
MAX_BLOCK_SIZE = int(os.environ.get("MAX_BLOCK_SIZE", "200"))
MAX_CANDIDATES_PER_ENTITY = MAX_BLOCK_SIZE

# Negative to positive sampling ratio for model training
NEGATIVE_SAMPLE_RATIO = int(os.environ.get("NEGATIVE_SAMPLE_RATIO", "5"))

# Stopword-like business tokens that are too common to block on alone
GENERIC_NAME_TOKENS = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "llc", "llp", "pvt", "private", "plc",
    "group", "holdings", "enterprises", "services", "solutions",
    "the", "and", "of", "&",
}

# ---------------------------------------------------------------------------
# Modeling & Threshold Tunables
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
VALIDATION_FRACTION = 0.2
F_BETA = 0.5

# Max Source-1 entities to sample for training & threshold tuning on large datasets.
# 100,000 entities produces ~1M candidate pairs, sufficient for GBDT convergence.
_max_train = os.environ.get("TRAIN_MAX_ENTITIES", "100000")
TRAIN_MAX_ENTITIES = int(_max_train) if _max_train and _max_train != "0" else None
