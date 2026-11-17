"""Stateful, sklearn-compatible preprocessing.

These components learn statistics (clip bounds, medians, scaling factors), so
they must be ``fit`` on the *training split only* and re-used unchanged for
validation, test and production scoring. Part 2 plugs ``build_preprocessor``
straight into a model ``Pipeline``.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Winsorise numeric columns at learned quantiles (NaNs are preserved).

    Args:
        lower_quantile: Lower clip quantile in ``[0, 1)``.
        upper_quantile: Upper clip quantile in ``(0, 1]``.
    """

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None) -> "OutlierClipper":  # noqa: N803 (sklearn convention)
        if not 0 <= self.lower_quantile < self.upper_quantile <= 1:
            raise ValueError("Require 0 <= lower_quantile < upper_quantile <= 1.")
        arr = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(arr, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.nanquantile(arr, self.upper_quantile, axis=0)
        self.n_features_in_ = arr.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X) -> np.ndarray:  # noqa: N803
        check_is_fitted(self, ["lower_bounds_", "upper_bounds_"])
        return np.clip(np.asarray(X, dtype=float), self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Return output feature names (identical to the inputs)."""
        check_is_fitted(self, ["lower_bounds_"])
        names = input_features if input_features is not None else getattr(self, "feature_names_in_", None)
        if names is None:
            names = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(names, dtype=object)


def build_preprocessor(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    *,
    scaler: str = "standard",
    clip_quantiles: tuple[float, float] = (0.01, 0.99),
    add_missing_indicators: bool = True,
) -> ColumnTransformer:
    """Create an unfitted preprocessing ``ColumnTransformer``.

    Numeric branch: quantile clipping -> median imputation (+ optional
    missing-value flags) -> scaling. Categorical branch: most-frequent
    imputation -> one-hot encoding (unknown categories are ignored at
    inference time, so new values never crash scoring).

    Args:
        numeric_columns: Numeric feature names.
        categorical_columns: Categorical feature names.
        scaler: ``"standard"``, ``"robust"`` or ``"none"``.
        clip_quantiles: ``(lower, upper)`` winsorisation quantiles.
        add_missing_indicators: Append ``missingindicator_*`` columns for
            numeric features that had gaps during ``fit`` (missingness is
            often informative in credit data).

    Returns:
        An unfitted ``ColumnTransformer`` producing a dense array.
    """
    scalers = {"standard": StandardScaler, "robust": RobustScaler}
    if scaler not in (*scalers, "none"):
        raise ValueError(f"scaler must be one of {[*scalers, 'none']}, got {scaler!r}")

    numeric_steps: list[tuple[str, object]] = [
        ("clipper", OutlierClipper(*clip_quantiles)),
        ("imputer", SimpleImputer(strategy="median", add_indicator=add_missing_indicators)),
    ]
    if scaler != "none":
        numeric_steps.append(("scaler", scalers[scaler]()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), list(numeric_columns)),
            ("categorical", Pipeline(categorical_steps), list(categorical_columns)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
