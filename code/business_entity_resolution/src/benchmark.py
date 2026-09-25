"""
benchmark.py - Scalability and Memory Profiling Benchmark
=========================================================

Runs an isolated end-to-end benchmark on a specified subset of entities:
  - Measures peak RAM usage (psutil)
  - Measures throughput (rows/sec, candidate-pairs/sec)
  - Reports average candidate pairs per entity
  - Reports disk usage (raw TSV, Parquet, free disk)
  - Produces empirical runtime and memory projections for 1M, 5M, 10M, 20M rows

Usage:
    python -m src.benchmark --rows 10000
    python -m src.benchmark --rows 100000
"""

import argparse
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import duckdb
import pandas as pd
import psutil

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.convert_to_parquet import convert_all
from src.blocking import DuckDBBlocker, candidate_stats
from src.features import build_feature_table
from src.model import load_model, train_classifier, predict_scores


class MemoryMonitor:
    """Background thread tracking peak resident memory usage (RSS)."""
    def __init__(self, interval=0.1):
        self.interval = interval
        self.peak_bytes = 0
        self.running = False
        self.process = psutil.Process()

    def start(self):
        self.running = True
        self.peak_bytes = self.process.memory_info().rss
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def _monitor(self):
        while self.running:
            try:
                rss = self.process.memory_info().rss
                if rss > self.peak_bytes:
                    self.peak_bytes = rss
            except Exception:
                break
            time.sleep(self.interval)

    def stop(self) -> float:
        self.running = False
        self.thread.join(timeout=1.0)
        return self.peak_bytes / (1024 ** 3)


def get_disk_report() -> dict:
    """Report storage footprints and available space."""
    free_bytes = shutil.disk_usage(config.REPO_ROOT).free
    
    def _dir_size(p: Path) -> float:
        if not p.exists():
            return 0.0
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 ** 2)

    raw_mb = _dir_size(config.REAL_DIR) if config.REAL_DIR.exists() else _dir_size(config.TRAIN_DIR)
    parquet_mb = _dir_size(config.PARQUET_DIR)
    out_mb = _dir_size(config.OUTPUT_DIR)
    
    return {
        "raw_mb": raw_mb,
        "parquet_mb": parquet_mb,
        "output_mb": out_mb,
        "free_gb": free_bytes / (1024 ** 3),
    }


def run_benchmark(n_rows: int = 10000, chunk_size: int = None):
    chunk_size = chunk_size or config.SOURCE1_CHUNK_SIZE
    print("=" * 72)
    print(f"  Amazon ML Challenge — Scalability Benchmark ({n_rows:,} entities)")
    print("=" * 72)

    disk_info = get_disk_report()
    print(f"Storage Status:")
    print(f"  Raw Dataset  : {disk_info['raw_mb']:,.1f} MB")
    print(f"  Parquet Cache: {disk_info['parquet_mb']:,.1f} MB")
    print(f"  Free Disk    : {disk_info['free_gb']:.1f} GB available")
    print(f"System Hardware:")
    print(f"  CPU Cores    : {psutil.cpu_count(logical=True)} logical cores")
    print(f"  Total RAM    : {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print("-" * 72)

    # 1. Ensure Parquet tables exist
    if not (config.PARQUET_TRAIN_SOURCE1.exists() and config.PARQUET_TRAIN_POOL.exists()):
        print(f"[INFO] Parquet tables not found. Running streaming conversion (up to {n_rows * 5:,} rows for benchmark)...")
        convert_all(force=False, max_rows=n_rows * 5, train_only=True)

    con = duckdb.connect()
    escaped_s1 = str(config.PARQUET_TRAIN_SOURCE1).replace("\\", "/")
    print(f"Loading {n_rows:,} Source-1 entities for benchmarking...")
    s1_df = con.execute(
        f"SELECT * FROM read_parquet('{escaped_s1}') LIMIT {n_rows}"
    ).df()
    con.close()

    print(f"Initializing DuckDBBlocker over {config.PARQUET_TRAIN_POOL.name}...")
    blocker = DuckDBBlocker(
        config.PARQUET_TRAIN_POOL,
        memory_limit=config.DUCKDB_MEMORY_LIMIT,
        threads=config.LIGHTGBM_THREADS,
    )

    # 2. Check/create sample model for scoring
    if config.MODEL_PATH.exists():
        model, tfidf = load_model(config.MODEL_PATH)
    else:
        print("[INFO] Quick-fitting benchmark model on first 500 records...")
        mini_pairs = blocker.block_chunk(s1_df.head(500), max_candidates=config.MAX_BLOCK_SIZE)
        mini_feats, tfidf = build_feature_table(mini_pairs, s1_df.head(500), blocker)
        mini_labels = [1 if i % 5 == 0 else 0 for i in range(len(mini_feats))]
        model = train_classifier(mini_feats, mini_labels)

    # 3. Benchmark Execution
    monitor = MemoryMonitor()
    monitor.start()

    total_pairs = 0
    t_start = time.time()
    t_block_total = 0.0
    t_feat_total = 0.0
    t_pred_total = 0.0

    n_chunks = (len(s1_df) + chunk_size - 1) // chunk_size
    print(f"\nRunning benchmark in {n_chunks} chunks of {chunk_size:,} entities...")

    for i in range(0, len(s1_df), chunk_size):
        chunk = s1_df.iloc[i : i + chunk_size]

        t0 = time.time()
        chunk_pairs = blocker.block_chunk(chunk, max_candidates=config.MAX_BLOCK_SIZE)
        t_block_total += time.time() - t0
        total_pairs += len(chunk_pairs)

        t0 = time.time()
        feats, _ = build_feature_table(chunk_pairs, chunk, blocker, tfidf_vectorizer=tfidf)
        t_feat_total += time.time() - t0

        t0 = time.time()
        _ = predict_scores(model, feats)
        t_pred_total += time.time() - t0

    total_elapsed = time.time() - t_start
    peak_ram_gb = monitor.stop()
    blocker.close()

    rows_per_sec = n_rows / total_elapsed if total_elapsed > 0 else 0
    pairs_per_sec = total_pairs / total_elapsed if total_elapsed > 0 else 0
    avg_cands = total_pairs / n_rows if n_rows > 0 else 0

    print("\n" + "=" * 72)
    print("  Benchmark Results")
    print("=" * 72)
    print(f"Entities Tested         : {n_rows:,}")
    print(f"Total Candidate Pairs   : {total_pairs:,}")
    print(f"Avg Candidates / Entity : {avg_cands:.2f}")
    print(f"Total Processing Time   : {total_elapsed:.2f} seconds")
    print(f"  - SQL Blocking Time   : {t_block_total:.2f}s ({(t_block_total/total_elapsed)*100:.1f}%)")
    print(f"  - Feature Gen Time    : {t_feat_total:.2f}s ({(t_feat_total/total_elapsed)*100:.1f}%)")
    print(f"  - Prediction Time     : {t_pred_total:.2f}s ({(t_pred_total/total_elapsed)*100:.1f}%)")
    print(f"Throughput              : {rows_per_sec:,.0f} entities/sec ({pairs_per_sec:,.0f} pairs/sec)")
    print(f"Peak Process RAM        : {peak_ram_gb:.2f} GB (System RAM: {psutil.virtual_memory().percent}% utilized)")
    print("-" * 72)

    # 4. Projections for Large Scale Datasets
    print("\n" + "=" * 72)
    print("  Empirical Scale Projections (Based on Measured Throughput)")
    print("=" * 72)
    print(f"{'Scale':<12} | {'Estimated Pairs':<18} | {'Estimated Time':<18} | {'Estimated RAM':<12}")
    print("-" * 68)

    scales = [1_000_000, 5_000_000, 10_000_000, 20_000_000]
    for scale in scales:
        est_pairs = int(scale * avg_cands)
        est_seconds = scale / rows_per_sec if rows_per_sec > 0 else 0
        if est_seconds < 3600:
            time_str = f"{est_seconds/60:.1f} minutes"
        else:
            time_str = f"{est_seconds/3600:.2f} hours"
        
        # RAM stays constant because of streaming chunk architecture!
        ram_str = f"~{peak_ram_gb:.1f} - {min(5.5, peak_ram_gb + 0.5):.1f} GB"
        print(f"{scale//1_000_000}M rows{'':<5} | {est_pairs:>15,} | {time_str:>16} | {ram_str:>10}")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile and benchmark the Entity Resolution pipeline")
    parser.add_argument("--rows", type=int, default=10000, help="Number of Source 1 entities to benchmark (e.g. 10000, 100000)")
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size for Source-1 processing")
    args = parser.parse_args()
    run_benchmark(n_rows=args.rows, chunk_size=args.chunk_size)
