import numpy as np
import pandas as pd
import pytest

from ml.config import ENGINEERED_CATEGORICAL_COLUMNS, ENGINEERED_NUMERIC_COLUMNS
from ml.features import FEATURE_DESCRIPTIONS, FeatureEngineer, engineer_features


def _row(**overrides) -> pd.DataFrame:
    base = {
        "age": 30, "annual_income": 60_000.0, "employment_type": "salaried",
        "employment_years": 5.0, "loan_amount": 12_000.0, "loan_term_months": 24,
        "monthly_debt_payment": 1_000.0, "credit_limit": 10_000.0,
        "revolving_balance": 8_000.0, "savings_balance": 30_000.0,
        "monthly_expenses": 2_500.0, "num_late_payments": 2,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_expected_columns_are_created(clean_df):
    out = engineer_features(clean_df)
    for col in (*ENGINEERED_NUMERIC_COLUMNS, *ENGINEERED_CATEGORICAL_COLUMNS):
        assert col in out.columns
    assert set(FEATURE_DESCRIPTIONS) == set(ENGINEERED_NUMERIC_COLUMNS) | set(ENGINEERED_CATEGORICAL_COLUMNS)


def test_ratio_values_are_correct():
    out = engineer_features(_row()).iloc[0]
    assert out["debt_to_income"] == pytest.approx(1_000 / 5_000)
    assert out["loan_to_income"] == pytest.approx(0.2)
    assert out["loan_payment_to_income"] == pytest.approx((12_000 / 24) / 5_000)
    assert out["expense_ratio"] == pytest.approx(0.5)
    assert out["savings_ratio"] == pytest.approx(0.5)
    assert out["savings_runway_months"] == pytest.approx(12)
    assert out["credit_utilization"] == pytest.approx(0.8)
    assert out["high_utilization_flag"] == 1.0
    assert out["has_late_payments"] == 1.0
    assert out["employment_stability_score"] == pytest.approx(0.5)
    assert out["age_group"] == "25-34"
    assert out["income_bucket"] == "mid"


def test_zero_denominators_give_nan_not_inf():
    out = engineer_features(_row(annual_income=0.0, credit_limit=0.0, monthly_expenses=0.0)).iloc[0]
    for col in ("debt_to_income", "loan_to_income", "credit_utilization", "savings_runway_months"):
        assert np.isnan(out[col])


def test_missing_inputs_propagate_as_nan():
    out = engineer_features(_row(num_late_payments=np.nan, employment_years=np.nan)).iloc[0]
    assert np.isnan(out["has_late_payments"])
    assert np.isnan(out["employment_stability_score"])


def test_unemployed_has_zero_stability():
    out = engineer_features(_row(employment_type="unemployed")).iloc[0]
    assert out["employment_stability_score"] == 0.0


def test_no_infinite_values_and_input_not_mutated(clean_df):
    before = clean_df.copy()
    out = engineer_features(clean_df)
    assert not np.isinf(out.select_dtypes("number").to_numpy()).any()
    pd.testing.assert_frame_equal(clean_df, before)


def test_feature_engineer_transformer_matches_function(clean_df):
    pd.testing.assert_frame_equal(FeatureEngineer().fit_transform(clean_df), engineer_features(clean_df))
