import json

import numpy as np
import pandas as pd
import pytest

from ml.artifacts import load_bundle
from ml.config import SEGMENT_SETTINGS, SegmentSettings
from ml.predict import CreditRiskPredictor
from ml.segmentation import assign_risk_segment, summarize_segments

GOOD_CUSTOMER = {
    "customer_id": "TEST001", "age": 45, "annual_income": 95_000, "employment_type": "salaried",
    "employment_years": 15, "loan_amount": 8_000, "loan_term_months": 36, "credit_score": 780,
    "monthly_debt_payment": 400, "credit_limit": 25_000, "revolving_balance": 2_000,
    "savings_balance": 60_000, "monthly_expenses": 3_000, "num_existing_loans": 1,
    "num_late_payments": 0, "repayment_history": "excellent", "home_ownership": "own",
}
RISKY_CUSTOMER = {
    "customer_id": "TEST002", "age": 22, "annual_income": 18_000, "employment_type": "unemployed",
    "employment_years": 0.2, "loan_amount": 25_000, "loan_term_months": 24, "credit_score": 420,
    "monthly_debt_payment": 900, "credit_limit": 3_000, "revolving_balance": 3_300,
    "savings_balance": 0, "monthly_expenses": 1_400, "num_existing_loans": 4,
    "num_late_payments": 8, "repayment_history": "poor", "home_ownership": "rent",
}


@pytest.fixture(scope="module")
def predictor(trained_run) -> CreditRiskPredictor:
    return CreditRiskPredictor.from_path(trained_run.artifact_paths["bundle"])


def test_predict_returns_probability_label_and_segment(predictor):
    out = predictor.predict(pd.DataFrame([GOOD_CUSTOMER, RISKY_CUSTOMER]))
    assert list(out.columns) == [
        "customer_id", "default_probability", "predicted_label", "risk_segment", "risk_label"
    ]
    assert out["default_probability"].between(0, 1).all()
    assert out["predicted_label"].isin([0, 1]).all()
    assert out["risk_segment"].isin(SEGMENT_SETTINGS.codes).all()
    assert out["risk_label"].isin(SEGMENT_SETTINGS.labels).all()


def test_risky_customer_scores_higher_than_good_customer(predictor):
    out = predictor.predict([GOOD_CUSTOMER, RISKY_CUSTOMER]).set_index("customer_id")
    assert out.loc["TEST002", "default_probability"] > out.loc["TEST001", "default_probability"]
    assert out.loc["TEST001", "risk_segment"] == "low"
    assert out.loc["TEST002", "risk_segment"] in {"medium", "high"}


def test_flagged_customer_is_never_low_risk(predictor):
    probs = np.linspace(0, 1, 101)
    seg = assign_risk_segment(probs, SEGMENT_SETTINGS, flag_threshold=predictor.bundle.decision_threshold)
    flagged = probs >= predictor.bundle.decision_threshold
    assert not (seg["risk_segment"][flagged] == "low").any()


def test_predict_one_is_json_serialisable(predictor):
    result = predictor.predict_one(GOOD_CUSTOMER)
    assert {"default_probability", "predicted_label", "risk_segment", "model_version"} <= set(result)
    json.dumps(result)


def test_handles_missing_values_unseen_categories_and_invalid_values(predictor):
    messy = {**GOOD_CUSTOMER, "credit_score": None, "savings_balance": np.nan,
             "home_ownership": "houseboat", "age": 150}
    out = predictor.predict(messy)
    assert len(out) == 1 and 0 <= out.loc[0, "default_probability"] <= 1


def test_predict_preserves_row_count_and_order(predictor):
    frame = pd.DataFrame([RISKY_CUSTOMER, GOOD_CUSTOMER, RISKY_CUSTOMER])
    out = predictor.predict(frame)
    assert out["customer_id"].tolist() == ["TEST002", "TEST001", "TEST002"]


def test_missing_required_column_raises(predictor):
    broken = {k: v for k, v in GOOD_CUSTOMER.items() if k != "credit_score"}
    with pytest.raises(ValueError, match="credit_score"):
        predictor.predict(broken)


def test_raw_dataset_rows_can_be_scored_end_to_end(predictor, trained_run):
    raw = pd.read_csv(trained_run.paths.raw_dataset).head(200)  # contains NaNs and bad values
    scored = predictor.predict(raw)
    assert len(scored) == 200 and scored["default_probability"].notna().all()


def test_bundle_roundtrip_keeps_decision_settings(trained_run):
    bundle = load_bundle(trained_run.artifact_paths["bundle"])
    assert bundle.model_name == trained_run.best_model_name
    assert bundle.decision_threshold == pytest.approx(trained_run.bundle.decision_threshold)
    assert bundle.segment_settings == SEGMENT_SETTINGS
    with pytest.raises(FileNotFoundError):
        load_bundle(trained_run.artifact_paths["bundle"].parent / "missing.joblib")


# ------------------------------------------------------------------- segmentation
def test_segment_boundaries_follow_config():
    s = SegmentSettings(low_max=0.2, high_min=0.5)
    out = assign_risk_segment(np.array([0.0, 0.199, 0.2, 0.499, 0.5, 1.0]), s)
    assert out["risk_segment"].tolist() == ["low", "low", "medium", "medium", "high", "high"]
    assert out["risk_label"].iloc[-1] == "High Risk"


def test_segment_validation_errors():
    with pytest.raises(ValueError):
        assign_risk_segment(np.array([0.5]), SegmentSettings(low_max=0.6, high_min=0.4))
    with pytest.raises(ValueError):
        assign_risk_segment(np.array([1.2]))
    with pytest.raises(ValueError):
        assign_risk_segment(np.array([np.nan]))


def test_segment_summary_is_monotonic_on_test(trained_run):
    best = next(e for e in trained_run.evaluations if e.model_name == trained_run.best_model_name)
    summary = summarize_segments(
        best.probabilities["test"], trained_run.splits.y_test, SEGMENT_SETTINGS, best.threshold
    ).set_index("risk_segment")
    assert summary["n_customers"].sum() == len(trained_run.splits.y_test)
    rates = summary["observed_default_rate"].dropna().tolist()
    assert rates == sorted(rates)
