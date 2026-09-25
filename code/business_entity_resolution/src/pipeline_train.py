"""
Training entry point.

Run from the `business_entity_resolution/` project root:

    python -m src.pipeline_train

Produces `models/matcher.joblib` (classifier + fitted TF-IDF vectorizer +
chosen decision threshold) and prints blocking-recall and validation-score
diagnostics along the way, so a bad blocking stage or a bad threshold shows
up before you ever touch the test set.
"""

import json
import sys
import time

from . import config
from .data_io import read_source, read_ground_truth, stack_candidate_pool
from .blocking import BlockIndex, generate_candidate_pairs, blocking_recall
from .labeling import label_candidate_pairs, entity_level_split
from .features import build_feature_table
from .model import train_classifier, predict_scores, save_model
from .threshold import search_threshold
from .evaluate import macro_f_beta, worst_entities


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr)


def main():
    _log("Loading training data...")
    source1 = read_source(config.TRAIN_SOURCE1)
    source2 = read_source(config.TRAIN_SOURCE2)
    source3 = read_source(config.TRAIN_SOURCE3)
    ground_truth = read_ground_truth(config.TRAIN_GROUND_TRUTH)
    candidate_pool = stack_candidate_pool(source2, source3)
    _log(
        f"source1={len(source1)} source2={len(source2)} "
        f"source3={len(source3)} ground_truth_rows={len(ground_truth)}"
    )

    # ------------------------------------------------------------------
    # 1. Blocking (over the FULL candidate pool - only Source 1 queries
    #    get split into train/validation, so validation candidates are
    #    generated exactly the way they will be at test time).
    # ------------------------------------------------------------------
    _log("Building blocking index over Source 2 + Source 3...")
    index = BlockIndex(candidate_pool)

    _log("Generating candidate pairs for all Source 1 training entities...")
    candidate_pairs = generate_candidate_pairs(source1, index)

    recall_report = blocking_recall(candidate_pairs, ground_truth)
    _log(f"Blocking recall report: {recall_report}")
    if recall_report["blocking_recall"] < 0.95:
        _log(
            "WARNING: blocking recall is below 0.95 - this caps your final "
            "F_0.5 no matter how good the classifier is. Widen blocking "
            "keys (e.g. add more phonetic variants, lower token generic-"
            "ness threshold) before trusting downstream numbers."
        )

    # ------------------------------------------------------------------
    # 2. Label pairs, split by ENTITY (never by pair) to avoid leakage.
    # ------------------------------------------------------------------
    labeled_pairs = label_candidate_pairs(candidate_pairs, ground_truth)
    train_ids, val_ids = entity_level_split(
        source1["entity_id"], config.VALIDATION_FRACTION, config.RANDOM_STATE
    )
    train_pairs = labeled_pairs[labeled_pairs["source1_entity_id"].isin(train_ids)]
    val_pairs = labeled_pairs[labeled_pairs["source1_entity_id"].isin(val_ids)]
    _log(
        f"Train entities={len(train_ids)} (pairs={len(train_pairs)}, "
        f"positives={train_pairs['label'].sum()}) | "
        f"Val entities={len(val_ids)} (pairs={len(val_pairs)}, "
        f"positives={val_pairs['label'].sum()})"
    )

    # ------------------------------------------------------------------
    # 3. Feature engineering (fit TF-IDF once, reuse for val + later
    #    inference so the feature space is consistent).
    # ------------------------------------------------------------------
    _log("Building training feature table...")
    train_source1 = source1[source1["entity_id"].isin(train_ids)]
    train_feats, tfidf_vectorizer = build_feature_table(
        train_pairs, train_source1, candidate_pool
    )

    _log("Building validation feature table...")
    val_source1 = source1[source1["entity_id"].isin(val_ids)]
    val_feats, _ = build_feature_table(
        val_pairs, val_source1, candidate_pool, tfidf_vectorizer=tfidf_vectorizer
    )

    # ------------------------------------------------------------------
    # 4. Train classifier.
    # ------------------------------------------------------------------
    _log("Training classifier...")
    model = train_classifier(train_feats, train_pairs["label"].values)

    # ------------------------------------------------------------------
    # 5. Score validation candidates, search the decision threshold
    #    directly against macro F_0.5 (singletons included).
    # ------------------------------------------------------------------
    _log("Scoring validation candidates...")
    val_feats = val_feats.copy()
    val_feats["score"] = predict_scores(model, val_feats)

    ground_truth_lookup = dict(
        zip(ground_truth["source1_entity_id"], ground_truth["matched_entity_ids"])
    )
    # Every val Source 1 entity needs a ground-truth entry, even singletons
    # (train_ground_truth.tsv already includes empty-match rows per the
    # README, but default to "" defensively).
    for s1_id in val_ids:
        ground_truth_lookup.setdefault(s1_id, "")

    best_threshold, best_score, grid = search_threshold(
        val_feats, val_ids, ground_truth_lookup
    )
    _log(f"Best threshold={best_threshold:.3f} -> macro F_0.5={best_score:.4f}")
    _log(f"Full threshold grid: {grid}")

    from .output import predictions_from_scores

    val_predictions = predictions_from_scores(val_feats, val_ids, best_threshold)
    eval_result = macro_f_beta(val_predictions, ground_truth_lookup, beta=config.F_BETA)
    _log("Worst-scoring validation entities (for error analysis):")
    for s1_id, score in worst_entities(eval_result, n=10):
        _log(f"  {s1_id}: score={score:.2f}")

    # ------------------------------------------------------------------
    # 6. Persist model + vectorizer + threshold.
    # ------------------------------------------------------------------
    save_model(model, tfidf_vectorizer, config.MODEL_PATH)
    threshold_path = config.MODEL_DIR / "threshold.json"
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    with open(threshold_path, "w") as f:
        json.dump(
            {"threshold": best_threshold, "validation_macro_f_beta": best_score}, f
        )
    _log(f"Saved model to {config.MODEL_PATH} and threshold to {threshold_path}")


if __name__ == "__main__":
    main()
