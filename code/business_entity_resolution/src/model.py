"""
The matching classifier.

Production recommendation: LightGBM or XGBoost (both MIT/Apache-2.0-licensed
gradient-boosted trees, well under the 8B-parameter cap, train on CPU in
minutes even on millions of rows). Example swap:

    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=400, num_leaves=63, learning_rate=0.05,
        class_weight="balanced", random_state=config.RANDOM_STATE,
    )

This module defaults to sklearn's HistGradientBoostingClassifier (BSD-3,
ships with scikit-learn, no extra install) so the pipeline is runnable
anywhere scikit-learn is installed, including offline sandboxes. It is
architecturally the same idea as LightGBM (histogram-based gradient
boosting) and is a reasonable stand-in for development; swap in LightGBM
for your real training run - it's what the license/performance envelope of
this challenge is built around.
"""

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config
from .features import FEATURE_COLUMNS


def train_classifier(feature_table, labels) -> HistGradientBoostingClassifier:
    """
    Parameters
    ----------
    feature_table : DataFrame with at least FEATURE_COLUMNS
    labels : array-like of 0/1

    Returns a fitted classifier.
    """
    X = feature_table[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = np.asarray(labels)

    # Candidate pairs are heavily imbalanced toward negatives (most blocked
    # candidates are not true matches) - class_weight handles that without
    # needing to manually resample.
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_depth=6,
        class_weight="balanced",
        random_state=config.RANDOM_STATE,
    )
    model.fit(X, y)
    return model


def predict_scores(model, feature_table) -> np.ndarray:
    """Return the model's predicted probability of a true match, per row."""
    X = feature_table[FEATURE_COLUMNS].to_numpy(dtype=float)
    return model.predict_proba(X)[:, 1]


def save_model(model, tfidf_vectorizer, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "tfidf": tfidf_vectorizer}, path)


def load_model(path):
    bundle = joblib.load(path)
    return bundle["model"], bundle["tfidf"]
