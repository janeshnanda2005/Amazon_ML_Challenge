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
import sys
import time

from . import config
from .data_io import read_source, stack_candidate_pool
from .blocking import BlockIndex, generate_candidate_pairs
from .features import build_feature_table
from .model import load_model, predict_scores
from .output import (
    predictions_from_scores,
    candidates_from_pairs,
    write_matching_results,
    write_candidate_pairs,
)


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr)


def main():
    _log("Loading test data...")
    source1 = read_source(config.TEST_SOURCE1)
    source2 = read_source(config.TEST_SOURCE2)
    source3 = read_source(config.TEST_SOURCE3)
    candidate_pool = stack_candidate_pool(source2, source3)
    _log(f"test source1={len(source1)} source2={len(source2)} source3={len(source3)}")

    _log("Loading trained model + vectorizer...")
    model, tfidf_vectorizer = load_model(config.MODEL_PATH)
    with open(config.MODEL_DIR / "threshold.json") as f:
        threshold = json.load(f)["threshold"]
    _log(f"Using decision threshold={threshold}")

    _log("Building blocking index over test Source 2 + Source 3...")
    index = BlockIndex(candidate_pool)

    _log("Generating candidate pairs for every test Source 1 entity...")
    candidate_pairs = generate_candidate_pairs(source1, index)
    _log(f"Generated {len(candidate_pairs)} candidate pairs.")

    _log("Building features and scoring...")
    feats, _ = build_feature_table(
        candidate_pairs, source1, candidate_pool, tfidf_vectorizer=tfidf_vectorizer
    )
    feats = feats.copy()
    feats["score"] = predict_scores(model, feats)

    all_source1_ids = source1["entity_id"].tolist()

    predictions = predictions_from_scores(feats, all_source1_ids, threshold)
    candidates = candidates_from_pairs(candidate_pairs, all_source1_ids)

    write_matching_results(predictions, config.MATCHING_RESULTS_PATH)
    write_candidate_pairs(candidates, config.CANDIDATE_PAIRS_PATH)
    _log(f"Wrote {config.MATCHING_RESULTS_PATH}")
    _log(f"Wrote {config.CANDIDATE_PAIRS_PATH}")
    _log(
        "Next: run utils/validate_submission.py against these two files "
        "before uploading matching_results.tsv to the leaderboard."
    )


if __name__ == "__main__":
    main()
