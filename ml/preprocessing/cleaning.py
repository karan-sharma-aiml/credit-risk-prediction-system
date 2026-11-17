"""Stateless data cleaning.

Everything here is *row-wise or rule-based*: it learns nothing from the data,
so it is safe to run on the full dataset before the train/test split.
Statistics-based steps (outlier bounds, imputation, scaling) are stateful and
live in ``ml.preprocessing.transformers`` so they can be fit on training data
only.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ml.config import (
    CATEGORICAL_COLUMNS,
    CATEGORY_LEVELS,
    ID_COL,
    NUMERIC_COLUMNS,
    TARGET_COL,
    TARGET_LABEL_MAP,
    VALID_RANGES,
)

logger = logging.getLogger(__name__)


@dataclass
class CleaningReport:
    """What the cleaning step changed (useful for audits and monitoring)."""

    rows_in: int = 0
    rows_out: int = 0
    exact_duplicates_removed: int = 0
    duplicate_ids_removed: int = 0
    rows_dropped_invalid_target: int = 0
    invalid_categories_nulled: dict[str, int] = field(default_factory=dict)
    out_of_range_nulled: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a plain ``dict`` representation."""
        return dict(self.__dict__)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case column names and convert spaces/dashes to underscores."""
    renamed = {c: re.sub(r"[\s\-]+", "_", str(c).strip().lower()) for c in df.columns}
    return df.rename(columns=renamed)


def coerce_dtypes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Force numeric columns to numbers and normalise categorical strings.

    Unparseable numbers become NaN. Categorical values are lower-cased,
    snake_cased and set to NaN when outside the allowed levels.

    Returns:
        The converted frame and a ``{column: n_values_nulled}`` map for
        categorical columns.
    """
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    nulled: dict[str, int] = {}
    for col, levels in CATEGORY_LEVELS.items():
        if col not in df.columns:
            continue
        text = df[col].astype("string").str.strip().str.lower().str.replace(r"[\s\-]+", "_", regex=True)
        before = int(text.notna().sum())
        text = text.where(text.isin(levels))
        lost = before - int(text.notna().sum())
        if lost:
            nulled[col] = lost
        df[col] = text.astype(object).where(text.notna(), np.nan)
    return df, nulled


def clean_target(df: pd.DataFrame, target_col: str = TARGET_COL) -> tuple[pd.DataFrame, int]:
    """Map target spellings to ``{0, 1}`` and drop rows without a valid label.

    Returns:
        The cleaned frame (target as ``int8``) and the number of dropped rows.
    """
    df = df.copy()
    labels = df[target_col].astype(str).str.strip().str.lower().map(TARGET_LABEL_MAP)
    valid = labels.notna()
    dropped = int((~valid).sum())
    if dropped:
        logger.warning("Dropping %d rows with missing/unrecognised target", dropped)
    df = df.loc[valid].copy()
    df[target_col] = labels.loc[valid].astype("int8")
    return df, dropped


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Drop exact duplicate rows, then repeated customer IDs (keep first).

    Returns:
        Deduplicated frame, number of exact duplicates, number of ID duplicates.
    """
    n0 = len(df)
    df = df.drop_duplicates(keep="first")
    exact = n0 - len(df)

    id_dupes = 0
    if ID_COL in df.columns:
        n1 = len(df)
        df = df.drop_duplicates(subset=[ID_COL], keep="first")
        id_dupes = n1 - len(df)
    return df.reset_index(drop=True), exact, id_dupes


def apply_domain_rules(
    df: pd.DataFrame,
    ranges: dict[str, tuple[float | None, float | None]] = VALID_RANGES,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Set values outside business-valid ranges to NaN.

    Returns:
        The frame and ``{column: n_values_nulled}`` for affected columns.
    """
    df = df.copy()
    nulled: dict[str, int] = {}
    for col, (low, high) in ranges.items():
        if col not in df.columns:
            continue
        bad = pd.Series(False, index=df.index)
        if low is not None:
            bad |= df[col] < low
        if high is not None:
            bad |= df[col] > high
        count = int(bad.sum())
        if count:
            df.loc[bad, col] = np.nan
            nulled[col] = count
    return df, nulled


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Run the full stateless cleaning sequence.

    Order matters: types first (so comparisons are valid), then the target,
    then de-duplication, then domain rules.

    Returns:
        Cleaned DataFrame and a :class:`CleaningReport`.
    """
    report = CleaningReport(rows_in=len(df))

    df = standardize_column_names(df)
    df, report.invalid_categories_nulled = coerce_dtypes(df)
    df, report.rows_dropped_invalid_target = clean_target(df)
    df, report.exact_duplicates_removed, report.duplicate_ids_removed = remove_duplicates(df)
    df, report.out_of_range_nulled = apply_domain_rules(df)

    report.rows_out = len(df)
    logger.info(
        "Cleaning: %d -> %d rows (dupes=%d, bad target=%d, out-of-range cells=%d)",
        report.rows_in, report.rows_out,
        report.exact_duplicates_removed + report.duplicate_ids_removed,
        report.rows_dropped_invalid_target,
        sum(report.out_of_range_nulled.values()),
    )
    return df, report


def clean_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Row-preserving cleaning for scoring new customers.

    Unlike :func:`clean_dataset` this never drops, re-orders or de-duplicates
    rows (every request must map to exactly one response row) and does not
    need a target. It applies the same type coercion and domain rules as
    training, so a value that would be nulled in training is nulled here too.

    Raises:
        ValueError: If a base feature column is entirely absent.
    """
    df = standardize_column_names(df)
    required = [c for c in (*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS) if c not in df.columns]
    if required:
        raise ValueError(f"Missing required input columns: {required}")
    df, _ = coerce_dtypes(df)
    df, _ = apply_domain_rules(df)
    return df
