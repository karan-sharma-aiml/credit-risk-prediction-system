"""Fake API client so dashboard pages can be exercised without a running backend."""

from __future__ import annotations

import pytest

from dashboard.api_client import APIError

RECENT = {
    "prediction_id": 7, "customer_id": "CUST_UI_1", "default_probability": 0.42, "predicted_label": 1,
    "risk_segment": "high", "risk_label": "High Risk", "model_version": "rf-test",
    "decision_threshold": 0.09, "created_at": "2026-11-15T10:00:00Z",
    "recommendation": {"action": "reject", "title": "Reject Loan", "message": "High default risk.",
                       "considerations": ["Debt-to-income ratio is above 43%."]},
}
HIST = {"bin_edges": [0.0, 10.0, 20.0], "counts": [3, 5]}
CUSTOMER = {"customer_id": "CUST_UI_1", "age": 40, "annual_income": 70000.0, "employment_type": "salaried",
            "employment_years": 8.0, "loan_amount": 12000.0, "loan_term_months": 36, "credit_score": 700,
            "monthly_debt_payment": 500.0, "credit_limit": 10000.0, "revolving_balance": 2000.0,
            "savings_balance": 9000.0, "monthly_expenses": 2500.0, "num_existing_loans": 1,
            "num_late_payments": 0, "repayment_history": "good", "home_ownership": "own",
            "is_default": None, "source_dataset": "api", "created_at": "2026-11-01T00:00:00Z",
            "updated_at": "2026-11-01T00:00:00Z"}


class FakeClient:
    """Implements the APIClient surface with canned data and records calls."""

    def __init__(self, empty: bool = False) -> None:
        self.empty = empty
        self.calls: list[tuple[str, object]] = []

    def health(self):
        return {"status": "ok", "model": "rf-test"}

    def model_info(self):
        return {"model_version": "rf-test", "decision_threshold": 0.09, "balanced_with_smote": True,
                "segment_cutoffs": {"low_max": 0.10, "high_min": 0.30},
                "validation_metrics": {"recall": 0.77, "roc_auc": 0.76}, "test_metrics": {"recall": 0.75, "roc_auc": 0.77}}

    def model_performance(self):
        return {"roc_curve": {"x": [0, 0.5, 1], "y": [0, 0.8, 1], "auc": 0.77},
                "pr_curve": {"x": [0, 0.5, 1], "y": [1, 0.4, 0.14], "auc": 0.33},
                "confusion_matrix": {"tn": 10, "fp": 2, "fn": 1, "tp": 4},
                "feature_importance": [{"feature": "credit_score", "importance": 0.2}, {"feature": "debt_to_income", "importance": 0.1}]}

    def dashboard(self):
        n = 0 if self.empty else 8
        return {
            "kpis": {"total_customers": n, "total_predictions": n, "average_default_probability": None if self.empty else 0.14,
                     "high_risk_share": None if self.empty else 0.1},
            "risk_distribution": [{"segment": s, "label": l, "count": c, "share": 0.0} for s, l, c in
                                  (("low", "Low Risk", 5), ("medium", "Medium Risk", 2), ("high", "High Risk", 1))],
            "income_distribution": {"bin_edges": [], "counts": []} if self.empty else HIST,
            "loan_distribution": HIST, "credit_score_distribution": HIST,
            "prediction_trend": [{"date": "2026-11-14", "count": 4, "avg_probability": 0.15}],
            "recent_predictions": [] if self.empty else [RECENT],
        }

    def list_customers(self, limit=50, offset=0, search=None):
        self.calls.append(("list_customers", search))
        return {"items": [CUSTOMER], "total": 1, "limit": limit, "offset": offset}

    def get_customer(self, customer_id):
        if customer_id != "CUST_UI_1":
            raise APIError(f"Customer '{customer_id}' not found.", 404)
        return CUSTOMER

    def create_customer(self, payload):
        self.calls.append(("create_customer", payload))
        return {**CUSTOMER, "customer_id": "CUST_NEW"}

    def update_customer(self, customer_id, payload):
        self.calls.append(("update_customer", (customer_id, payload)))
        return CUSTOMER

    def delete_customer(self, customer_id):
        self.calls.append(("delete_customer", customer_id))

    def predict(self, payload):
        self.calls.append(("predict", payload))
        return {**RECENT, "history": {"previous_segment": "low", "probability_change": 0.3, "input_snapshot": payload}}

    def list_predictions(self, limit=50, offset=0, **filters):
        self.calls.append(("list_predictions", filters))
        return {"items": [RECENT], "total": 1, "limit": limit, "offset": offset}

    def get_prediction(self, prediction_id):
        return {**RECENT, "request_id": "req-1",
                "history": {"input_snapshot": {"credit_score": 700}, "previous_segment": None, "segment_changed": False,
                            "probability_change": None}}


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr("dashboard.api_client.get_client", lambda: client)
    return client
