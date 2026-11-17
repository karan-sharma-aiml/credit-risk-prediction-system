"""Stratified train / validation / test splitting.

Fitting of *anything* (preprocessor, SMOTE, models, thresholds' search space)
may only use the training split. The validation split is used for threshold
and model selection; the test split is touched only for final reporting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.config import RANDOM_STATE, SPLIT_SETTINGS, SplitSettings

logger = logging.getLogger(__name__)


@dataclass
class DataSplits:
    """Container for the three stratified splits (indices are preserved)."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

    def class_distribution(self) -> pd.DataFrame:
        """Rows, defaults and default rate per split."""
        rows = []
        for name, y in (("train", self.y_train), ("validation", self.y_val), ("test", self.y_test)):
            rows.append({
                "split": name,
                "n_rows": len(y),
                "n_default": int(y.sum()),
                "n_repaid": int((y == 0).sum()),
                "default_rate": round(float(y.mean()), 4),
            })
        return pd.DataFrame(rows)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    settings: SplitSettings = SPLIT_SETTINGS,
    seed: int = RANDOM_STATE,
) -> DataSplits:
    """Create stratified train / validation / test splits.

    Args:
        X: Feature frame.
        y: Binary target aligned with ``X``.
        settings: Split fractions (relative to the full dataset).
        seed: Random seed.

    Raises:
        ValueError: If fractions are invalid or a class is too rare to stratify.
    """
    if not 0 < settings.test_size + settings.validation_size < 1:
        raise ValueError("test_size + validation_size must be in (0, 1).")
    if y.value_counts().min() < 3:
        raise ValueError("Each class needs at least 3 samples for a stratified 3-way split.")

    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=settings.test_size, stratify=y, random_state=seed
    )
    val_fraction = settings.validation_size / (1.0 - settings.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_fraction, stratify=y_rest, random_state=seed
    )
    splits = DataSplits(X_train, X_val, X_test, y_train, y_val, y_test)
    logger.info("Split sizes -> train=%d, validation=%d, test=%d", len(y_train), len(y_val), len(y_test))
    return splits
