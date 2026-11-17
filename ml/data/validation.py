"""Schema validation and data-quality inspection (no mutation of the input)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd
from pandas.api.types import is_numeric_dtype

from ml.config import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COL,
)

logger = logging.getLogger(__name__)


class SchemaValidationError(ValueError):
    """Raised when the dataset does not match the expected schema."""


@dataclass
class DataQualityReport:
    """Snapshot of dataset health, serialisable to JSON."""

    n_rows: int
    n_columns: int
    memory_mb: float
    n_duplicate_rows: int
    n_missing_cells: int
    missing_by_column: dict[str, int]
    dtype_issues: list[str]
    unexpected_columns: list[str]
    target_distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a plain ``dict`` representation."""
        return dict(self.__dict__)


def validate_columns(
    df: pd.DataFrame,
    required: Iterable[str] = REQUIRED_COLUMNS,
) -> list[str]:
    """Check that every required column exists.

    Args:
        df: Dataset to validate.
        required: Column names that must be present.

    Returns:
        Columns present in ``df`` but not part of the schema (informational).

    Raises:
        SchemaValidationError: If any required column is missing.
    """
    required = list(required)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaValidationError(f"Missing required columns: {missing}")
    unexpected = [c for c in df.columns if c not in required]
    if unexpected:
        logger.warning("Unexpected columns will be carried through: %s", unexpected)
    return unexpected


def inspect_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Compare actual dtypes with the schema's expectation.

    Returns:
        DataFrame with ``column``, ``dtype``, ``expected`` and ``ok`` columns.
    """
    rows = []
    for col in df.columns:
        if col in NUMERIC_COLUMNS or col == TARGET_COL:
            expected, ok = "numeric", is_numeric_dtype(df[col])
        elif col in CATEGORICAL_COLUMNS:
            expected, ok = "categorical", not is_numeric_dtype(df[col])
        else:
            expected, ok = "any", True
        rows.append({"column": col, "dtype": str(df[col].dtype), "expected": expected, "ok": ok})
    return pd.DataFrame(rows)


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing counts and percentages (columns with gaps first)."""
    counts = df.isna().sum()
    report = pd.DataFrame(
        {"missing_count": counts, "missing_pct": (counts / max(len(df), 1) * 100).round(3)}
    )
    return report.sort_values("missing_count", ascending=False)


def count_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> int:
    """Number of duplicated rows (every repeat after the first occurrence)."""
    return int(df.duplicated(subset=subset).sum())


def summarize_dataset(df: pd.DataFrame) -> DataQualityReport:
    """Build a :class:`DataQualityReport` for ``df``."""
    dtype_table = inspect_dtypes(df)
    dtype_issues = [
        f"{r.column}: dtype {r.dtype}, expected {r.expected}"
        for r in dtype_table.itertuples()
        if not r.ok
    ]
    missing = df.isna().sum()
    target_dist: dict[str, int] = {}
    if TARGET_COL in df.columns:
        target_dist = {str(k): int(v) for k, v in df[TARGET_COL].value_counts(dropna=False).items()}

    report = DataQualityReport(
        n_rows=len(df),
        n_columns=df.shape[1],
        memory_mb=round(float(df.memory_usage(deep=True).sum() / 1024**2), 3),
        n_duplicate_rows=count_duplicates(df),
        n_missing_cells=int(missing.sum()),
        missing_by_column={c: int(v) for c, v in missing.items() if v > 0},
        dtype_issues=dtype_issues,
        unexpected_columns=[c for c in df.columns if c not in REQUIRED_COLUMNS],
        target_distribution=target_dist,
    )
    logger.info(
        "Data quality: %d rows, %d duplicates, %d missing cells, %d dtype issues",
        report.n_rows, report.n_duplicate_rows, report.n_missing_cells, len(dtype_issues),
    )
    return report
