import pytest

from tests.backend.payloads import GOOD_CUSTOMER, RISKY_CUSTOMER


def test_predict_returns_probability_label_segment_recommendation(client):
    response = client.post("/predict", json=GOOD_CUSTOMER)
    assert response.status_code == 201
    body = response.json()
    assert 0 <= body["default_probability"] <= 1
    assert body["predicted_label"] in (0, 1)
    assert body["risk_segment"] in ("low", "medium", "high")
    assert body["risk_label"] in ("Low Risk", "Medium Risk", "High Risk")
    assert body["recommendation"]["action"] in ("approve", "manual_review", "reject")
    assert body["model_version"].startswith("random_forest")
    assert response.headers["X-Request-ID"] == body["request_id"]


def test_api_reuses_credit_risk_predictor_exactly(client, predictor):
    body = client.post("/predict", json=GOOD_CUSTOMER).json()
    direct = predictor.predict_one({**GOOD_CUSTOMER, "customer_id": body["customer_id"]})
    assert body["default_probability"] == pytest.approx(direct["default_probability"], abs=1e-6)
    assert body["risk_segment"] == direct["risk_segment"]
    assert body["predicted_label"] == direct["predicted_label"]
    assert body["model_version"] == direct["model_version"]


def test_prediction_saves_customer_prediction_and_audit_snapshot(client):
    body = client.post("/predict", json=GOOD_CUSTOMER).json()
    customer = client.get(f"/customers/{body['customer_id']}").json()
    assert customer["credit_score"] == GOOD_CUSTOMER["credit_score"] and customer["source_dataset"] == "api"
    detail = client.get(f"/predictions/{body['prediction_id']}").json()
    assert detail["history"]["input_snapshot"]["credit_score"] == 780
    assert detail["history"]["previous_prediction_id"] is None


def test_recommendation_follows_segment(client):
    good = client.post("/predict", json=GOOD_CUSTOMER).json()
    risky = client.post("/predict", json=RISKY_CUSTOMER).json()
    assert good["risk_segment"] == "low" and good["recommendation"]["action"] == "approve"
    assert risky["risk_segment"] in ("medium", "high")
    assert risky["recommendation"]["action"] == {"medium": "manual_review", "high": "reject"}[risky["risk_segment"]]
    assert risky["default_probability"] > good["default_probability"]
    assert risky["recommendation"]["considerations"]  # high DTI / utilisation / late payments ...


def test_rescoring_existing_customer_updates_data_and_tracks_history(client):
    first = client.post("/predict", json={**GOOD_CUSTOMER, "customer_id": "CUST_RESCORE"}).json()
    second = client.post("/predict", json={**RISKY_CUSTOMER, "customer_id": "CUST_RESCORE"}).json()
    assert second["customer_id"] == "CUST_RESCORE"
    assert client.get("/customers/CUST_RESCORE").json()["credit_score"] == RISKY_CUSTOMER["credit_score"]
    history = client.get(f"/predictions/{second['prediction_id']}").json()["history"]
    assert history["previous_prediction_id"] == first["prediction_id"]
    assert history["previous_segment"] == "low" and history["segment_changed"] is True
    assert history["probability_change"] == pytest.approx(second["default_probability"] - first["default_probability"], abs=1e-5)


def test_optional_fields_may_be_omitted(client):
    minimal = {k: v for k, v in GOOD_CUSTOMER.items()
               if k not in ("credit_limit", "revolving_balance", "repayment_history", "home_ownership",
                            "num_late_payments", "num_existing_loans", "employment_years")}
    response = client.post("/predict", json=minimal)
    assert response.status_code == 201 and 0 <= response.json()["default_probability"] <= 1


def test_invalid_payload_is_rejected_and_nothing_is_stored(client):
    assert client.post("/predict", json={**GOOD_CUSTOMER, "credit_score": 1000}).status_code == 422
    assert client.get("/customers").json()["total"] == 0 and client.get("/predictions").json()["total"] == 0


def test_inference_failure_returns_500_and_rolls_back(client, monkeypatch):
    def broken(_record):
        raise RuntimeError("boom")

    monkeypatch.setattr(client.app.state.model_context.predictor, "predict_one", broken)
    response = client.post("/predict", json={**GOOD_CUSTOMER, "customer_id": "CUST_FAIL"})
    assert response.status_code == 500 and response.json()["error"]["code"] == "prediction_failed"
    assert client.get("/customers/CUST_FAIL").status_code == 404  # customer upsert rolled back
