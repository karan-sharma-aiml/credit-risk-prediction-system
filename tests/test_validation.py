import pytest

from ml.config import REQUIRED_COLUMNS
from ml.data import (
    SchemaValidationError,
    count_duplicates,
    inspect_dtypes,
    missing_value_report,
    summarize_dataset,
    validate_columns,
)


def test_validate_columns_passes_on_full_schema(raw_df):
    assert validate_columns(raw_df) == []


def test_validate_columns_raises_on_missing(raw_df):
    with pytest.raises(SchemaValidationError, match="credit_score"):
        validate_columns(raw_df.drop(columns=["credit_score"]))


def test_validate_columns_reports_unexpected(raw_df):
    assert validate_columns(raw_df.assign(extra=1)) == ["extra"]


def test_duplicates_and_missing_detected(raw_df):
    assert count_duplicates(raw_df) > 0
    report = missing_value_report(raw_df)
    assert report["missing_count"].iloc[0] > 0


def test_inspect_dtypes_flags_string_in_numeric_column(raw_df):
    broken = raw_df.assign(age=raw_df["age"].astype(str))
    table = inspect_dtypes(broken).set_index("column")
    assert not table.loc["age", "ok"]


def test_summary_counts_match(raw_df):
    report = summarize_dataset(raw_df)
    assert report.n_rows == len(raw_df)
    assert report.n_missing_cells == int(raw_df.isna().sum().sum())
    assert set(REQUIRED_COLUMNS) >= set(report.missing_by_column)
