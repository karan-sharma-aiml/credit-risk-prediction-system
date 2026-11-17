from datetime import datetime, timezone

from tests.backend.payloads import GOOD_CUSTOMER, RISKY_CUSTOMER


def _seed(client):
    ids = []
    for i, payload in enumerate([GOOD_CUSTOMER, RISKY_CUSTOMER, GOOD_CUSTOMER]):
        ids.append(client.post("/predict", json={**payload, "customer_id": f"CUST_H{i}"}).json())
    return ids


def test_prediction_history_list_filters_and_pagination(client):
    _seed(client)
    page = client.get("/predictions").json()
    assert page["total"] == 3 and page["items"][0]["customer_id"] == "CUST_H2"  # newest first
    assert client.get("/predictions", params={"customer_id": "CUST_H1"}).json()["total"] == 1
    low = client.get("/predictions", params={"risk_segment": "low"}).json()
    assert low["total"] >= 2 and all(p["risk_segment"] == "low" for p in low["items"])
    assert len(client.get("/predictions", params={"limit": 1, "offset": 1}).json()["items"]) == 1
    assert client.get("/predictions", params={"risk_segment": "nonsense"}).status_code == 422
    assert client.get("/predictions", params={"date_from": "2999-01-01T00:00:00"}).json()["total"] == 0


def test_get_prediction_detail_and_404(client):
    created = _seed(client)[0]
    detail = client.get(f"/predictions/{created['prediction_id']}").json()
    assert detail["prediction_id"] == created["prediction_id"] and detail["history"] is not None
    assert client.get("/predictions/999999").status_code == 404


def test_deleting_customer_removes_predictions(client):
    created = _seed(client)[1]
    assert client.delete("/customers/CUST_H1").status_code == 204
    assert client.get(f"/predictions/{created['prediction_id']}").status_code == 404
    assert client.get("/predictions", params={"customer_id": "CUST_H1"}).json()["total"] == 0


def test_dashboard_empty_state(client):
    body = client.get("/dashboard").json()
    assert body["kpis"]["total_predictions"] == 0 and body["kpis"]["average_default_probability"] is None
    assert [b["count"] for b in body["risk_distribution"]] == [0, 0, 0]
    assert body["income_distribution"] == {"bin_edges": [], "counts": []}
    assert body["prediction_trend"] == [] and body["recent_predictions"] == []


def test_dashboard_aggregates_after_predictions(client):
    _seed(client)
    body = client.get("/dashboard").json()
    assert body["kpis"]["total_customers"] == 3 and body["kpis"]["total_predictions"] == 3
    assert 0 < body["kpis"]["average_default_probability"] < 1
    assert sum(b["count"] for b in body["risk_distribution"]) == 3
    assert [b["segment"] for b in body["risk_distribution"]] == ["low", "medium", "high"]
    assert sum(body["income_distribution"]["counts"]) == 3
    assert sum(body["credit_score_distribution"]["counts"]) == 3
    assert body["prediction_trend"][0]["count"] == 3 and len(body["recent_predictions"]) == 3
    assert body["prediction_trend"][0]["date"] == datetime.now(timezone.utc).date().isoformat()  # UTC day


def test_dashboard_risk_distribution_uses_latest_prediction_per_customer(client):
    client.post("/predict", json={**GOOD_CUSTOMER, "customer_id": "CUST_X"})
    client.post("/predict", json={**RISKY_CUSTOMER, "customer_id": "CUST_X"})
    body = client.get("/dashboard").json()
    assert body["kpis"]["total_predictions"] == 2
    assert sum(b["count"] for b in body["risk_distribution"]) == 1
    assert next(b for b in body["risk_distribution"] if b["segment"] == "low")["count"] == 0


def test_model_info_and_performance(client):
    info = client.get("/model-info").json()
    assert info["segment_cutoffs"] == {"low_max": 0.10, "high_min": 0.30}
    assert info["segment_labels"]["high"] == "High Risk" and info["n_features"] > 20
    assert 0.5 < info["test_metrics"]["roc_auc"] < 1 and "confusion_matrix" not in info["test_metrics"]
    perf = client.get("/model-info/performance").json()
    assert perf["roc_curve"]["auc"] > 0.5 and len(perf["roc_curve"]["x"]) <= 200
    assert set(perf["confusion_matrix"]) == {"tn", "fp", "fn", "tp"}
    assert 1 <= len(perf["feature_importance"]) <= 15
    assert perf["feature_importance"][0]["importance"] >= perf["feature_importance"][-1]["importance"]
