"""Model registry: every classifier behind one interface.

Each entry is an unfitted scikit-learn-compatible estimator, so training,
tuning and evaluation code never special-cases an algorithm. XGBoost and
LightGBM are included automatically when installed.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from ml.config import RANDOM_STATE

logger = logging.getLogger(__name__)


def get_model_registry(seed: int = RANDOM_STATE) -> dict[str, BaseEstimator]:
    """Return ``{name: unfitted estimator}`` with sensible, regularised defaults."""
    registry: dict[str, BaseEstimator] = {
        "logistic_regression": LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
        "decision_tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=30, random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=10, n_jobs=-1, random_state=seed
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=seed
        ),
    }
    if importlib.util.find_spec("xgboost"):
        from xgboost import XGBClassifier

        registry["xgboost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1, random_state=seed,
        )
    else:
        logger.info("xgboost not installed; skipping")
    if importlib.util.find_spec("lightgbm"):
        from lightgbm import LGBMClassifier

        registry["lightgbm"] = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8,
            colsample_bytree=0.8, n_jobs=-1, random_state=seed, verbose=-1,
        )
    else:
        logger.info("lightgbm not installed; skipping")
    return registry


#: Small search spaces for ``RandomizedSearchCV`` (parameter names without
#: the pipeline's ``model__`` prefix).
PARAM_DISTRIBUTIONS: dict[str, dict[str, list[Any]]] = {
    "logistic_regression": {"C": [0.01, 0.1, 1.0, 10.0]},
    "decision_tree": {"max_depth": [3, 4, 6, 8], "min_samples_leaf": [10, 30, 60]},
    "random_forest": {
        "n_estimators": [200, 300, 500],
        "max_depth": [6, 10, 14],
        "min_samples_leaf": [5, 10, 20],
    },
    "gradient_boosting": {
        "n_estimators": [100, 150, 250],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [2, 3, 4],
    },
    "xgboost": {
        "n_estimators": [200, 300, 500],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [3, 4, 6],
    },
    "lightgbm": {
        "n_estimators": [200, 300, 500],
        "learning_rate": [0.03, 0.05, 0.1],
        "num_leaves": [15, 31, 63],
    },
}
