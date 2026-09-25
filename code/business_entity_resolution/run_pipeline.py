#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io, sys
# Force UTF-8 on Windows consoles so box-drawing chars don't crash cp1252
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
"""
run_pipeline.py - Full end-to-end pipeline automation
======================================================

This script orchestrates every stage of the Business Entity Resolution
pipeline in the correct order, with structured logging, elapsed-time
reporting, and clear error messages so you know exactly what failed and why.

Usage (from the `business_entity_resolution/` directory):
    python run_pipeline.py [--synthetic] [--skip-train] [--skip-validate]
                           [--check-ids] [--lightgbm] [--rapidfuzz]

Flags:
    --synthetic      Generate a fresh synthetic dataset before running
                     (required when real data has NOT yet been placed in
                     dataset/train/ and dataset/test/).
    --skip-train     Skip training; load an existing model from models/.
                     Useful when iterating on threshold or predict only.
    --skip-validate  Skip the official validate_submission.py check.
    --check-ids      Pass --check-ids to the validator (also validates that
                     matched IDs actually exist in the test source files).
    --lightgbm       Patch model.py to use LightGBM at runtime (requires
                     `pip install lightgbm`).
    --rapidfuzz      Patch similarity.py to use rapidfuzz at runtime
                     (requires `pip install rapidfuzz`).
"""

import argparse
import importlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

# ── colour codes (stripped on non-TTY) ────────────────────────────────────────
_IS_TTY = sys.stdout.isatty()

def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _IS_TTY else text

GREEN  = lambda t: _c("32;1", t)
YELLOW = lambda t: _c("33;1", t)
RED    = lambda t: _c("31;1", t)
CYAN   = lambda t: _c("36;1", t)
BOLD   = lambda t: _c("1", t)
DIM    = lambda t: _c("2", t)

# ── logging helpers ────────────────────────────────────────────────────────────
_RUN_START = time.time()

def log(msg, *, level="INFO"):
    elapsed = time.time() - _RUN_START
    tag = {
        "INFO":  GREEN("[INFO]"),
        "WARN":  YELLOW("[WARN]"),
        "ERROR": RED("[ERROR]"),
        "STEP":  CYAN("[STEP]"),
    }.get(level, f"[{level}]")
    ts = time.strftime("%H:%M:%S")
    print(f"{DIM(ts)}  {tag}  {msg}  {DIM(f'+{elapsed:.1f}s')}", flush=True)

def step(title):
    width = 72
    bar   = "-" * width
    print(f"\n{CYAN(bar)}")
    print(f"{CYAN('|')}  {BOLD(title)}")
    print(f"{CYAN(bar)}")

def banner(title, subtitle=""):
    width = 72
    bar   = "=" * width
    print(f"\n{CYAN(bar)}")
    print(f"{CYAN('|')}  {BOLD(title)}")
    if subtitle:
        print(f"{CYAN('|')}  {DIM(subtitle)}")
    print(f"{CYAN(bar)}\n")

def success(msg):
    print(f"\n  {GREEN('[OK]')}  {GREEN(msg)}\n")

def fail(msg):
    print(f"\n  {RED('[FAIL]')}  {RED(msg)}\n")


# ── path constants ─────────────────────────────────────────────────────────────
HERE         = Path(__file__).resolve().parent          # business_entity_resolution/
# Find repo root
REPO_ROOT    = HERE.parent.parent if (HERE.parent.parent / "utils").exists() else HERE.parent
UTILS_DIR    = REPO_ROOT / "utils"
SRC          = HERE / "src"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from src import config

TRAIN_DIR    = config.TRAIN_DIR
TEST_DIR     = config.TEST_DIR
OUTPUT_DIR   = config.OUTPUT_DIR
MODEL_DIR    = config.MODEL_DIR
SYNTHETIC_PY = HERE / "make_synthetic_data.py"
VALIDATOR_PY = UTILS_DIR / "validate_submission.py"

def _rel(p: Path) -> str:
    for base in (REPO_ROOT, HERE):
        try:
            return str(p.relative_to(base))
        except ValueError:
            pass
    return str(p)


# ── optional backend patches ───────────────────────────────────────────────────
def _patch_lightgbm():
    """Monkey-patch src.model to use LightGBM if available."""
    try:
        import lightgbm as lgb  # noqa: F401
    except ImportError:
        log("lightgbm not installed – pip install lightgbm first", level="WARN")
        return False

    from src import model as _model, config as _config
    from src.features import FEATURE_COLUMNS
    import numpy as np

    def train_classifier(feature_table, labels):
        import lightgbm as lgb
        import numpy as np
        X = feature_table[FEATURE_COLUMNS].to_numpy(dtype=float)
        y = np.asarray(labels)
        m = lgb.LGBMClassifier(
            n_estimators=400, num_leaves=63, learning_rate=0.05,
            class_weight="balanced", random_state=_config.RANDOM_STATE,
            n_jobs=-1, verbose=-1,
        )
        m.fit(X, y)
        return m

    _model.train_classifier = train_classifier
    log("LightGBM backend active", level="INFO")
    return True


def _patch_rapidfuzz():
    """Monkey-patch src.similarity to use rapidfuzz if available."""
    try:
        from rapidfuzz.distance import Levenshtein, JaroWinkler  # noqa: F401
    except ImportError:
        log("rapidfuzz not installed – pip install rapidfuzz first", level="WARN")
        return False

    from src import similarity as _sim
    from rapidfuzz.distance import Levenshtein, JaroWinkler

    def lev_ratio(a, b):
        return Levenshtein.normalized_similarity(a, b)

    def jw(a, b, prefix_weight=0.1):
        return JaroWinkler.similarity(a, b)

    _sim.levenshtein_ratio = lev_ratio
    _sim.jaro_winkler       = jw
    log("rapidfuzz backend active (Levenshtein + Jaro-Winkler)", level="INFO")
    return True


# ── stage runners ──────────────────────────────────────────────────────────────
def stage_check_env():
    step("Stage 0 — Environment check")
    import pandas as pd, numpy as np, sklearn, joblib  # noqa
    log(f"Python  {sys.version.split()[0]}")
    log(f"pandas  {pd.__version__}")
    log(f"numpy   {np.__version__}")
    log(f"sklearn {sklearn.__version__}")
    log(f"joblib  {joblib.__version__}")

    optional = {}
    for pkg in ("lightgbm", "rapidfuzz"):
        try:
            mod = importlib.import_module(pkg)
            optional[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            optional[pkg] = "NOT installed (optional)"
    for pkg, ver in optional.items():
        log(f"{pkg:10s} {ver}", level="INFO")
    success("Environment OK")


def stage_synthetic():
    step("Stage 1 — Generate synthetic dataset")
    log("Running make_synthetic_data.py ...")
    result = subprocess.run(
        [sys.executable, str(SYNTHETIC_PY)],
        capture_output=True, text=True, cwd=str(HERE),
    )
    print(result.stdout)
    if result.returncode != 0:
        fail("Synthetic data generation failed")
        print(result.stderr)
        sys.exit(1)
    success("Synthetic dataset written to dataset/train/ and dataset/test/")


def stage_check_data():
    step("Stage 2 — Verify data files present")
    required = {
        "train": [
            TRAIN_DIR / "train_source1.tsv",
            TRAIN_DIR / "train_source2.tsv",
            TRAIN_DIR / "train_source3.tsv",
            TRAIN_DIR / "train_ground_truth.tsv",
        ],
        "test": [
            TEST_DIR / "test_source1.tsv",
            TEST_DIR / "test_source2.tsv",
            TEST_DIR / "test_source3.tsv",
        ],
    }
    missing = []
    for split, paths in required.items():
        for p in paths:
            if p.exists():
                sz = p.stat().st_size
                log(f"  FOUND  {_rel(p)}  ({sz:,} bytes)")
            else:
                log(f"  MISSING  {_rel(p)}", level="WARN")
                missing.append(p)

    if missing:
        fail(f"{len(missing)} required file(s) missing. "
             "Either place your real data there or re-run with --synthetic.")
        sys.exit(1)
    success("All required data files present")


def stage_train(args):
    step("Stage 3 — Training (blocking → labeling → features → model → threshold)")
    t0 = time.time()
    # Add src to path so imports work when called as a plain script
    sys.path.insert(0, str(HERE))

    if args.lightgbm:
        _patch_lightgbm()
    if args.rapidfuzz:
        _patch_rapidfuzz()

    # Reload modules fresh (important when --skip-train is later toggled)
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("src."):
            del sys.modules[mod_name]

    from src.pipeline_train import main as train_main
    train_main()

    elapsed = time.time() - t0
    success(f"Training complete in {elapsed:.1f}s")

    # Read back saved threshold for reporting
    threshold_path = MODEL_DIR / "threshold.json"
    if threshold_path.exists():
        info = json.loads(threshold_path.read_text())
        log(f"Saved threshold    : {info['threshold']:.4f}")
        log(f"Val macro F_0.5    : {info['validation_macro_f_beta']:.4f}")


def stage_predict(args):
    step("Stage 4 — Inference (blocking → features → score → write TSVs)")
    t0 = time.time()
    sys.path.insert(0, str(HERE))

    if args.lightgbm:
        _patch_lightgbm()
    if args.rapidfuzz:
        _patch_rapidfuzz()

    # Reload fresh so patched backends are seen
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("src."):
            del sys.modules[mod_name]

    from src.pipeline_predict import main as predict_main
    predict_main()

    elapsed = time.time() - t0
    success(f"Inference complete in {elapsed:.1f}s")

    # Quick size sanity-check on outputs
    for fname in ("matching_results.tsv", "candidate_pairs.tsv"):
        p = OUTPUT_DIR / fname
        if p.exists():
            rows = sum(1 for _ in open(p, encoding="utf-8")) - 1  # minus header
            log(f"  {fname}  →  {rows:,} data rows  ({p.stat().st_size:,} bytes)")
        else:
            log(f"  {fname} NOT found", level="WARN")


def stage_validate(args):
    step("Stage 5 — Official submission validator")
    if not VALIDATOR_PY.exists():
        log(f"Validator not found at {VALIDATOR_PY}", level="WARN")
        return

    cmd = [
        sys.executable, str(VALIDATOR_PY),
        "--matching", str(OUTPUT_DIR / "matching_results.tsv"),
        "--candidate", str(OUTPUT_DIR / "candidate_pairs.tsv"),
        "--test-dir", str(TEST_DIR),
    ]
    if args.check_ids:
        cmd.append("--check-ids")

    log(f"Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HERE))
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode == 0:
        success("Validator: PASS – safe to submit")
    else:
        fail("Validator: FAIL – fix the listed issues before submitting")
        sys.exit(1)


def stage_summary():
    step("Pipeline Summary")
    outputs = [
        OUTPUT_DIR / "matching_results.tsv",
        OUTPUT_DIR / "candidate_pairs.tsv",
        MODEL_DIR  / "matcher.joblib",
        MODEL_DIR  / "threshold.json",
    ]
    print()
    for p in outputs:
        exists = p.exists()
        mark   = GREEN("✔") if exists else RED("✗")
        size   = f"  ({p.stat().st_size:,} bytes)" if exists else "  MISSING"
        print(f"  {mark}  {p.relative_to(HERE)}{size}")
    print()


# ── main ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Run the full Business Entity Resolution pipeline end-to-end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--synthetic",      action="store_true",
                   help="Generate fresh synthetic data before running")
    p.add_argument("--convert-parquet", action="store_true",
                   help="Force rebuild Parquet cache from raw TSV data")
    p.add_argument("--benchmark",      action="store_true",
                   help="Run memory and scalability benchmark without running full pipeline")
    p.add_argument("--benchmark-rows", type=int, default=10000,
                   help="Number of entities for benchmark mode (default: 10000)")
    p.add_argument("--resume",         action="store_true",
                   help="Resume prediction from last saved checkpoint")
    p.add_argument("--skip-train",     action="store_true",
                   help="Skip training; load existing model from models/")
    p.add_argument("--skip-validate",  action="store_true",
                   help="Skip validate_submission.py after writing TSVs")
    p.add_argument("--check-ids",      action="store_true",
                   help="Pass --check-ids to the validator")
    p.add_argument("--lightgbm",       action="store_true",
                   help="Use LightGBM classifier (requires pip install lightgbm)")
    p.add_argument("--rapidfuzz",      action="store_true",
                   help="Use rapidfuzz string metrics (requires pip install rapidfuzz)")
    return p.parse_args()


def main():
    args = parse_args()

    # Benchmark mode early-exit
    if args.benchmark:
        from src.benchmark import run_benchmark
        run_benchmark(n_rows=args.benchmark_rows)
        return

    # Force parquet conversion early-exit
    if args.convert_parquet:
        from src.convert_to_parquet import convert_all
        convert_all(force=True)
        return

    banner(
        "Business Entity Resolution Pipeline",
        "DuckDB + Parquet → Blocking → Features → LightGBM → Incremental TSVs",
    )

    total_t0 = time.time()
    try:
        stage_check_env()

        if args.synthetic:
            stage_synthetic()

        stage_check_data()

        # Auto-ensure Parquet tables are ready
        from src.convert_to_parquet import convert_all
        if not (config.PARQUET_TRAIN_POOL.exists() and config.PARQUET_TEST_POOL.exists()):
            log("Parquet tables missing in data/parquet/. Running initial streaming conversion...")
            convert_all(force=False)

        if not args.skip_train:
            stage_train(args)
        else:
            log("--skip-train: skipping training stage", level="WARN")
            if not (MODEL_DIR / "matcher.joblib").exists():
                fail("No model found at models/matcher.joblib. "
                     "Remove --skip-train and run training first.")
                sys.exit(1)

        stage_predict(args)

        if not args.skip_validate:
            stage_validate(args)
        else:
            log("--skip-validate: skipping official validator", level="WARN")

        stage_summary()

        total_elapsed = time.time() - total_t0
        banner(
            f"ALL STAGES PASSED  ({total_elapsed:.1f}s total)",
            "Output files are in output/  —  ready to zip and submit.",
        )

    except SystemExit:
        raise
    except Exception:
        fail("Unexpected error – full traceback below:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
