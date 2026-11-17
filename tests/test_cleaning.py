import numpy as np
import pandas as pd

from ml.config import TARGET_COL, VALID_RANGES
from ml.preprocessing import clean_dataset, clean_target, remove_duplicates


def test_clean_dataset_removes_duplicates_and_bad_target(raw_df):
    cleaned, report = clean_dataset(raw_df)
    assert not cleaned.duplicated().any()
    assert cleaned["customer_id"].is_unique
    assert cleaned[TARGET_COL].isin([0, 1]).all()
    assert report.rows_in == len(raw_df)
    assert report.rows_out == len(cleaned) < report.rows_in


def test_out_of_range_values_become_nan(clean_df):
    low, high = VALID_RANGES["age"]
    assert clean_df["age"].dropna().between(low, high).all()
    assert (clean_df["savings_balance"].dropna() >= 0).all()


def test_clean_target_maps_label_variants():
    df = pd.DataFrame({TARGET_COL: ["Yes", "no", "1", 0, "maybe", np.nan]})
    cleaned, dropped = clean_target(df)
    assert cleaned[TARGET_COL].tolist() == [1, 0, 1, 0]
    assert dropped == 2


def test_remove_duplicates_keeps_first():
    df = pd.DataFrame({"customer_id": ["a", "a", "b"], "x": [1, 1, 2]})
    out, exact, id_dupes = remove_duplicates(df)
    assert len(out) == 2 and exact == 1 and id_dupes == 0


def test_cleaning_is_idempotent(raw_df):
    once, _ = clean_dataset(raw_df)
    twice, report = clean_dataset(once)
    assert len(once) == len(twice)
    assert report.exact_duplicates_removed == 0
    assert report.out_of_range_nulled == {}
