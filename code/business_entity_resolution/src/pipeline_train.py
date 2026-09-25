"""
Training entry point.

Run from the `business_entity_resolution/` project root:

    python -m src.pipeline_train

Architecture:
  - Uses DuckDBBlocker over precomputed Parquet candidate pool
  - Never loads 20M-row datasets into pandas RAM
  - Evaluates blocking recall and candidate distributions (avg, median, p95, p99, max)
  - Negative sampling: retains all true positives + samples hard negatives from blocking
  - Entity-level train/validation split (no leakage)
  - Trains LightGBM classifier with multithreading
  - Searches optimal threshold against Macro F0.5 (singletons included)
  - Persists model + vectorizer + threshold to disk
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
from .blocking import DuckDBBlocker, BlockIndex, generate_candidate_pairs, blocking_recall, candidate_stats
from .data_io import read_ground_truth
from .labeling import label_candidate_pairs, entity_level_split
from .features import build_feature_table
from .model import train_classifier, predict_scores, save_model
from .threshold import search_threshold
from .evaluate import macro_f_beta, worst_entities


def _get_ram_gb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 3)


def _log(msg):
    ts = time.strftime("%H:%M:%S")
    ram = _get_ram_gb()
    print(f"[{ts}] [RAM: {ram:.2f} GB] {msg}", flush=True)


def load_or_convert_data():
    """Ensure Parquet files exist, or trigger streaming conversion."""
    if not (config.PARQUET_TRAIN_SOURCE1.exists() and config.PARQUET_TRAIN_POOL.exists() and config.PARQUET_TRAIN_GROUND_TRUTH.exists()):
        _log("Parquet datasets not found in data/parquet/. Initiating streaming conversion from raw TSV...")
        convert_all(force=False)
    else:
        _log("Found existing Parquet tables in data/parquet/train/.")


def main():
    load_or_convert_data()

    con = duckdb.connect()
    con.execute(f"PRAGMA max_memory = '{config.DUCKDB_MEMORY_LIMIT}';")
    con.execute(f"PRAGMA threads = {config.LIGHTGBM_THREADS};")

    _log("Loading training entities and ground truth via DuckDB...")
    escaped_s1 = str(config.PARQUET_TRAIN_SOURCE1).replace("\\", "/")
    escaped_gt = str(config.PARQUET_TRAIN_GROUND_TRUTH).replace("\\", "/")

    total_s1 = con.execute(f"SELECT count(*) FROM read_parquet('{escaped_s1}')").fetchone()[0]
    _log(f"Total Source-1 training entities in Parquet: {total_s1:,}")

    # Load Source 1 sample for training if configured
    if config.TRAIN_MAX_ENTITIES and total_s1 > config.TRAIN_MAX_ENTITIES:
        _log(f"Sampling {config.TRAIN_MAX_ENTITIES:,} Source 1 entities for training & validation (controlled memory)...")
        source1 = con.execute(
            f"SELECT * FROM read_parquet('{escaped_s1}') USING SAMPLE {config.TRAIN_MAX_ENTITIES} (reservoir, 42)"
        ).df()
    else:
        source1 = con.execute(f"SELECT * FROM read_parquet('{escaped_s1}')").df()

    # Load ground truth for these entities
    con.register("sampled_s1", source1[["entity_id"]])
    ground_truth = con.execute(
        f"""
        SELECT gt.source1_entity_id, gt.matched_entity_ids 
        FROM read_parquet('{escaped_gt}') gt
        JOIN sampled_s1 s ON gt.source1_entity_id = s.entity_id
        """
    ).df()
    con.unregister("sampled_s1")
    con.close()

    _log(f"Loaded source1={len(source1):,} entities and ground_truth={len(ground_truth):,} rows.")

    # ------------------------------------------------------------------
    # 1. SQL Blocking via DuckDB over Parquet pool
    # ------------------------------------------------------------------
    _log(f"Initializing DuckDBBlocker over {config.PARQUET_TRAIN_POOL.name}...")
    blocker = DuckDBBlocker(
        config.PARQUET_TRAIN_POOL,
        memory_limit=config.DUCKDB_MEMORY_LIMIT,
        threads=config.LIGHTGBM_THREADS,
    )

    _log(f"Generating candidate pairs in chunks of {config.SOURCE1_CHUNK_SIZE:,} entities...")
    candidate_pairs_list = []
    t_block_start = time.time()
    n_chunks = (len(source1) + config.SOURCE1_CHUNK_SIZE - 1) // config.SOURCE1_CHUNK_SIZE

    for idx, start_i in enumerate(range(0, len(source1), config.SOURCE1_CHUNK_SIZE)):
        end_i = min(start_i + config.SOURCE1_CHUNK_SIZE, len(source1))
        chunk = source1.iloc[start_i:end_i]
        chunk_pairs = blocker.block_chunk(chunk, max_candidates=config.MAX_BLOCK_SIZE)
        candidate_pairs_list.append(chunk_pairs)
        
        if (idx + 1) % max(1, n_chunks // 5) == 0 or (idx + 1) == n_chunks:
            elapsed = time.time() - t_block_start
            speed = end_i / elapsed if elapsed > 0 else 0
            _log(f"  Chunk {idx+1}/{n_chunks}: Processed {end_i:,}/{len(source1):,} entities ({speed:.0f} entities/s)")

    candidate_pairs = pd.concat(candidate_pairs_list, ignore_index=True)
    _log(f"Candidate generation complete: {len(candidate_pairs):,} total pairs generated in {time.time()-t_block_start:.1f}s.")

    # Diagnostic statistics
    stats = candidate_stats(candidate_pairs, len(source1))
    _log(
        f"Candidate Distribution: avg={stats['avg']:.2f}, median={stats['median']:.1f}, "
        f"p95={stats['p95']:.1f}, p99={stats['p99']:.1f}, max={stats['max']}, total={stats['total_pairs']:,}"
    )

    recall_report = blocking_recall(candidate_pairs, ground_truth)
    _log(f"Blocking recall report: {recall_report}")
    if recall_report["blocking_recall"] < 0.95:
        _log("WARNING: Blocking recall is below 0.95. Candidate reduction or blocking keys may need tuning.")

    # ------------------------------------------------------------------
    # 2. Label pairs & Entity-level Split
    # ------------------------------------------------------------------
    _log("Labeling candidate pairs against ground truth...")
    labeled_pairs = label_candidate_pairs(candidate_pairs, ground_truth)
    
    train_ids, val_ids = entity_level_split(
        source1["entity_id"], config.VALIDATION_FRACTION, config.RANDOM_STATE
    )
    raw_train_pairs = labeled_pairs[labeled_pairs["source1_entity_id"].isin(train_ids)]
    val_pairs = labeled_pairs[labeled_pairs["source1_entity_id"].isin(val_ids)]

    # Negative sampling for training to prevent exploding negative pairs
    pos_train = raw_train_pairs[raw_train_pairs["label"] == 1]
    neg_train = raw_train_pairs[raw_train_pairs["label"] == 0]
    target_neg_count = len(pos_train) * config.NEGATIVE_SAMPLE_RATIO

    if len(neg_train) > target_neg_count:
        _log(f"Sampling {target_neg_count:,} hard negatives from {len(neg_train):,} total negatives (ratio {config.NEGATIVE_SAMPLE_RATIO}:1)...")
        sampled_neg = neg_train.sample(target_neg_count, random_state=config.RANDOM_STATE)
        train_pairs = pd.concat([pos_train, sampled_neg], ignore_index=True).sample(frac=1.0, random_state=config.RANDOM_STATE).reset_index(drop=True)
    else:
        train_pairs = raw_train_pairs.reset_index(drop=True)

    _log(
        f"Train dataset: entities={len(train_ids):,}, pairs={len(train_pairs):,} (pos={train_pairs['label'].sum():,}, neg={len(train_pairs)-train_pairs['label'].sum():,}) | "
        f"Val dataset: entities={len(val_ids):,}, pairs={len(val_pairs):,} (pos={val_pairs['label'].sum():,})"
    )

    # ------------------------------------------------------------------
    # 3. Feature Engineering with TF-IDF Vectorizer
    # ------------------------------------------------------------------
    _log("Building training features (using RapidFuzz C backend)...")
    train_source1 = source1[source1["entity_id"].isin(train_ids)]
    t_feat = time.time()
    train_feats, tfidf_vectorizer = build_feature_table(
        train_pairs, train_source1, blocker
    )
    _log(f"Built training feature matrix ({len(train_feats):,} rows) in {time.time()-t_feat:.1f}s.")

    _log("Building validation features...")
    val_source1 = source1[source1["entity_id"].isin(val_ids)]
    t_val_feat = time.time()
    val_feats, _ = build_feature_table(
        val_pairs, val_source1, blocker, tfidf_vectorizer=tfidf_vectorizer
    )
    _log(f"Built validation feature matrix ({len(val_feats):,} rows) in {time.time()-t_val_feat:.1f}s.")

    # ------------------------------------------------------------------
    # 4. Train Classifier
    # ------------------------------------------------------------------
    _log(f"Training LightGBM classifier with {config.LIGHTGBM_THREADS} threads...")
    t_train = time.time()
    model = train_classifier(train_feats, train_pairs["label"].values)
    _log(f"Classifier trained successfully in {time.time()-t_train:.1f}s.")

    # ------------------------------------------------------------------
    # 5. Threshold Optimization on Macro F0.5
    # ------------------------------------------------------------------
    _log("Scoring validation candidate pairs...")
    val_feats = val_feats.copy()
    val_feats["score"] = predict_scores(model, val_feats)

    ground_truth_lookup = dict(
        zip(ground_truth["source1_entity_id"], ground_truth["matched_entity_ids"])
    )
    for s1_id in val_ids:
        ground_truth_lookup.setdefault(s1_id, "")

    _log("Searching decision cutoff directly on Macro F0.5 with singletons...")
    best_threshold, best_score, grid = search_threshold(
        val_feats, val_ids, ground_truth_lookup
    )
    _log(f"Optimal Threshold: {best_threshold:.4f} -> Validation Macro F0.5: {best_score:.4f}")

    from .output import predictions_from_scores
    val_predictions = predictions_from_scores(val_feats, val_ids, best_threshold)
    eval_result = macro_f_beta(val_predictions, ground_truth_lookup, beta=config.F_BETA)
    _log("Worst-scoring validation entities:")
    for s1_id, score in worst_entities(eval_result, n=5):
        _log(f"  {s1_id}: score={score:.2f}")

    # ------------------------------------------------------------------
    # 6. Save Model Artifacts
    # ------------------------------------------------------------------
    save_model(model, tfidf_vectorizer, config.MODEL_PATH)
    config.THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.THRESHOLD_PATH, "w") as f:
        json.dump(
            {
                "threshold": float(best_threshold),
                "validation_macro_f_beta": float(best_score),
                "threads": config.LIGHTGBM_THREADS,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=2,
        )
    _log(f"Saved trained model to {config.MODEL_PATH} and threshold to {config.THRESHOLD_PATH}")
    blocker.close()


if __name__ == "__main__":
    main()
