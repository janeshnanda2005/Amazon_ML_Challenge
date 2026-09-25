"""
Inference entry point - produces matching_results.tsv and candidate_pairs.tsv.

Architecture:
  - Streaming, disk-backed inference using DuckDBBlocker over Parquet candidate pool
  - Never loads 20M-row datasets into pandas RAM
  - Source-1 processed in configurable chunks (default 2,000 entities)
  - Features generated and scored per batch, then immediately written to disk and freed
  - Restartable: stores progress checkpoints; can resume seamlessly if interrupted
  - Real-time monitoring: RAM usage, throughput, candidates/entity, and elapsed times
"""

import json
import os
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd
import psutil

from . import config
from .convert_to_parquet import convert_all
from .blocking import DuckDBBlocker, BlockIndex, generate_candidate_pairs
from .features import build_feature_table
from .model import load_model, predict_scores
from .output import (
    predictions_from_scores,
    candidates_from_pairs,
    append_results_tsv,
)


def _get_ram_gb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 3)


def _log(msg):
    ts = time.strftime("%H:%M:%S")
    ram = _get_ram_gb()
    print(f"[{ts}] [RAM: {ram:.2f} GB] {msg}", flush=True)


def load_checkpoint() -> dict:
    if config.CHECKPOINT_PATH.exists():
        try:
            with open(config.CHECKPOINT_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_checkpoint(last_completed_idx: int, total_pairs: int):
    config.CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = config.CHECKPOINT_PATH.with_suffix(".tmp")
    with open(temp_path, "w") as f:
        json.dump(
            {
                "last_completed_idx": last_completed_idx,
                "total_pairs_generated": total_pairs,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=2,
        )
    # Atomic rename to avoid corruption
    if temp_path.exists():
        temp_path.replace(config.CHECKPOINT_PATH)


def main():
    # Ensure Parquet tables exist
    if not (config.PARQUET_TEST_SOURCE1.exists() and config.PARQUET_TEST_POOL.exists()):
        _log("Test Parquet tables not found. Starting streaming conversion from raw TSV...")
        convert_all(force=False)

    _log("Loading trained matching model + vectorizer...")
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {config.MODEL_PATH}. Run training first.")
    model, tfidf_vectorizer = load_model(config.MODEL_PATH)

    threshold = 0.5
    if config.THRESHOLD_PATH.exists():
        with open(config.THRESHOLD_PATH) as f:
            threshold = json.load(f)["threshold"]
    _log(f"Using tuned decision cutoff threshold = {threshold:.4f}")

    # Checkpoint check for restartability
    checkpoint = load_checkpoint()
    start_entity_idx = checkpoint.get("last_completed_idx", 0)
    total_pairs_accumulated = checkpoint.get("total_pairs_generated", 0)

    is_resume = (start_entity_idx > 0) and config.MATCHING_RESULTS_PATH.exists() and config.CANDIDATE_PAIRS_PATH.exists()
    if is_resume:
        _log(f"Resuming prediction from entity index {start_entity_idx:,} (already generated {total_pairs_accumulated:,} pairs)...")
    else:
        start_entity_idx = 0
        total_pairs_accumulated = 0
        config.MATCHING_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        config.CANDIDATE_PAIRS_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Reset output files if starting fresh
        if config.MATCHING_RESULTS_PATH.exists():
            config.MATCHING_RESULTS_PATH.unlink()
        if config.CANDIDATE_PAIRS_PATH.exists():
            config.CANDIDATE_PAIRS_PATH.unlink()

    # Initialize DuckDB connection & blocker
    _log(f"Connecting DuckDB to candidate pool: {config.PARQUET_TEST_POOL.name}...")
    blocker = DuckDBBlocker(
        config.PARQUET_TEST_POOL,
        memory_limit=config.DUCKDB_MEMORY_LIMIT,
        threads=config.LIGHTGBM_THREADS,
    )

    con = duckdb.connect()
    escaped_s1 = str(config.PARQUET_TEST_SOURCE1).replace("\\", "/")
    total_test_entities = con.execute(f"SELECT count(*) FROM read_parquet('{escaped_s1}')").fetchone()[0]
    _log(f"Total test Source-1 entities to predict: {total_test_entities:,}")

    chunk_size = config.SOURCE1_CHUNK_SIZE
    n_chunks = (total_test_entities + chunk_size - 1) // chunk_size
    current_chunk_idx = start_entity_idx // chunk_size

    _log(f"Processing in streaming chunks of {chunk_size:,} entities ({n_chunks} total chunks)...")

    overall_start = time.time()
    current_idx = start_entity_idx

    while current_idx < total_test_entities:
        chunk_start_time = time.time()
        current_chunk_idx += 1
        limit_n = min(chunk_size, total_test_entities - current_idx)

        # Stream chunk of Source 1 from Parquet
        s1_chunk = con.execute(
            f"SELECT * FROM read_parquet('{escaped_s1}') LIMIT {limit_n} OFFSET {current_idx}"
        ).df()

        # Stage 1: Candidate generation
        t0 = time.time()
        chunk_pairs = blocker.block_chunk(s1_chunk, max_candidates=config.MAX_BLOCK_SIZE)
        t_blocking = time.time() - t0
        n_pairs = len(chunk_pairs)
        total_pairs_accumulated += n_pairs
        avg_cand = n_pairs / len(s1_chunk) if len(s1_chunk) > 0 else 0.0

        # Stage 2: Feature extraction
        t0 = time.time()
        if n_pairs > 0:
            feats, _ = build_feature_table(
                chunk_pairs, s1_chunk, blocker, tfidf_vectorizer=tfidf_vectorizer
            )
            feats = feats.copy()
            # Stage 3: Model scoring
            t_score0 = time.time()
            feats["score"] = predict_scores(model, feats)
            t_predict = time.time() - t_score0
        else:
            feats = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "score"])
            t_predict = 0.0
        t_features = time.time() - t0 - t_predict

        # Stage 4: Formatting & Incremental Output
        chunk_s1_ids = s1_chunk["entity_id"].tolist()
        chunk_preds = predictions_from_scores(feats, chunk_s1_ids, threshold)
        chunk_cands = candidates_from_pairs(chunk_pairs, chunk_s1_ids)

        write_header = (current_idx == 0)
        append_results_tsv(
            chunk_preds, config.MATCHING_RESULTS_PATH, "source1_entity_id", "matched_entity_ids", write_header=write_header
        )
        append_results_tsv(
            chunk_cands, config.CANDIDATE_PAIRS_PATH, "source1_entity_id", "candidate_entity_ids", write_header=write_header
        )

        current_idx += len(s1_chunk)
        save_checkpoint(current_idx, total_pairs_accumulated)

        # Monitoring
        chunk_duration = time.time() - chunk_start_time
        total_elapsed = time.time() - overall_start
        pct = (current_idx / total_test_entities) * 100
        speed_entities = current_idx / total_elapsed if total_elapsed > 0 else 0
        speed_pairs = total_pairs_accumulated / total_elapsed if total_elapsed > 0 else 0

        _log(
            f"Chunk {current_chunk_idx}/{n_chunks} ({pct:.1f}%): "
            f"S1={current_idx:,}/{total_test_entities:,} | "
            f"Cands={n_pairs:,} (avg {avg_cand:.1f}/ent) | "
            f"Times: block={t_blocking:.2f}s feat={t_features:.2f}s pred={t_predict:.2f}s | "
            f"Speed={speed_entities:.0f} ent/s ({speed_pairs:.0f} pairs/s)"
        )

        # Explicit memory cleanup for current batch
        del s1_chunk, chunk_pairs, feats, chunk_preds, chunk_cands

    con.close()
    blocker.close()

    total_time = time.time() - overall_start
    _log(
        f"Inference completed in {total_time:.1f}s! "
        f"Processed {total_test_entities:,} entities, {total_pairs_accumulated:,} candidate pairs."
    )
    _log(f"Matching results -> {config.MATCHING_RESULTS_PATH}")
    _log(f"Candidate pairs  -> {config.CANDIDATE_PAIRS_PATH}")

    # Remove checkpoint on clean completion
    if config.CHECKPOINT_PATH.exists():
        config.CHECKPOINT_PATH.unlink()


if __name__ == "__main__":
    main()
