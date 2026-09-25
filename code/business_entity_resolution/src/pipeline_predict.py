"""
Inference entry point - produces the two files scored/audited by the
challenge.

Run from the `business_entity_resolution/` project root, after
`pipeline_train.py` has produced `models/matcher.joblib`:

    python -m src.pipeline_predict

Writes:
    output/matching_results.tsv   (scored on the leaderboard)
    output/candidate_pairs.tsv    (blocking set fed to the model)
"""

import json
import os
import sys
import time
import pandas as pd

from . import config
from .data_io import read_source, stack_candidate_pool
from .blocking import BlockIndex, generate_candidate_pairs
from .features import build_feature_table
from .model import load_model, predict_scores
from .output import (
    predictions_from_scores,
    candidates_from_pairs,
    append_results_tsv,
)


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr)


def main():
    _log("Loading test data...")
    source1 = read_source(config.TEST_SOURCE1)
    source2 = read_source(config.TEST_SOURCE2)
    source3 = read_source(config.TEST_SOURCE3)
    candidate_pool = stack_candidate_pool(source2, source3)
    _log(f"test source1={len(source1):,} source2={len(source2):,} source3={len(source3):,}")

    _log("Loading trained model + vectorizer...")
    model, tfidf_vectorizer = load_model(config.MODEL_PATH)
    with open(config.MODEL_DIR / "threshold.json") as f:
        threshold = json.load(f)["threshold"]
    _log(f"Using decision threshold={threshold:.4f}")

    _log("Building blocking index over test Source 2 + Source 3...")
    index = BlockIndex(candidate_pool)

    # Chunked inference to guarantee low memory footprint (< 2 GB)
    chunk_size = int(os.environ.get("PREDICT_CHUNK_SIZE", 50000))
    total_entities = len(source1)
    _log(f"Running inference in chunks of {chunk_size:,} entities...")

    config.MATCHING_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.CANDIDATE_PAIRS_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_pairs = 0
    start_time = time.time()

    for start_idx in range(0, total_entities, chunk_size):
        end_idx = min(start_idx + chunk_size, total_entities)
        s1_chunk = source1.iloc[start_idx:end_idx]
        is_first = (start_idx == 0)

        chunk_pairs = generate_candidate_pairs(s1_chunk, index)
        total_pairs += len(chunk_pairs)

        if len(chunk_pairs) > 0:
            feats, _ = build_feature_table(
                chunk_pairs, s1_chunk, candidate_pool, tfidf_vectorizer=tfidf_vectorizer
            )
            feats = feats.copy()
            feats["score"] = predict_scores(model, feats)
        else:
            feats = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "score"])

        chunk_s1_ids = s1_chunk["entity_id"].tolist()
        chunk_preds = predictions_from_scores(feats, chunk_s1_ids, threshold)
        chunk_cands = candidates_from_pairs(chunk_pairs, chunk_s1_ids)

        append_results_tsv(
            chunk_preds, config.MATCHING_RESULTS_PATH, "source1_entity_id", "matched_entity_ids", write_header=is_first
        )
        append_results_tsv(
            chunk_cands, config.CANDIDATE_PAIRS_PATH, "source1_entity_id", "candidate_entity_ids", write_header=is_first
        )

        pct = (end_idx / total_entities) * 100
        elapsed = time.time() - start_time
        _log(f"Progress: {end_idx:,}/{total_entities:,} ({pct:.1f}%) entities | Total pairs: {total_pairs:,} | Elapsed: {elapsed:.1f}s")

    _log(f"Inference complete! Wrote {config.MATCHING_RESULTS_PATH} and {config.CANDIDATE_PAIRS_PATH}")
    _log(
        "Next: run utils/validate_submission.py against these two files "
        "before uploading matching_results.tsv to the leaderboard."
    )


if __name__ == "__main__":
    main()
