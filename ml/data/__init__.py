"""Data access: generation, loading and validation."""

from ml.data.generator import generate_synthetic_credit_data
from ml.data.loader import get_or_create_raw_dataset, load_dataset
from ml.data.validation import (
    DataQualityReport,
    SchemaValidationError,
    count_duplicates,
    inspect_dtypes,
    missing_value_report,
    summarize_dataset,
    validate_columns,
)

__all__ = [
    "DataQualityReport",
    "SchemaValidationError",
    "count_duplicates",
    "generate_synthetic_credit_data",
    "get_or_create_raw_dataset",
    "inspect_dtypes",
    "load_dataset",
    "missing_value_report",
    "summarize_dataset",
    "validate_columns",
]
