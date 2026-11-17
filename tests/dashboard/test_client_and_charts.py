import pytest
import requests

from dashboard import charts
from dashboard.api_client import APIClient, APIError
from dashboard.config import normalize_base_url
from tests.dashboard.conftest import FakeClient, HIST


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code, self._body, self.ok = status, body, 200 <= status < 300

    def json(self):
        if self._body is None:
            raise ValueError
        return self._body


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.last = response, error, None

    def request(self, method, url, **kwargs):
        self.last = (method, url, kwargs)
        if self.error:
            raise self.error
        return self.response


def test_client_returns_json_and_builds_urls():
    session = FakeSession(FakeResponse(200, {"items": []}))
    client = APIClient("http://api:8000", session=session)
    assert client.list_customers(limit=5, offset=10, search="AB") == {"items": []}
    method, url, kwargs = session.last
    assert (method, url) == ("GET", "http://api:8000/customers") and kwargs["params"] == {"limit": 5, "offset": 10, "search": "AB"}


def test_client_maps_error_envelope_and_204_and_connection_errors():
    err = {"error": {"code": "validation_error", "message": "Request validation failed.", "details": [{"field": "age", "message": "bad"}]}}
    with pytest.raises(APIError) as info:
        APIClient("http://x", session=FakeSession(FakeResponse(422, err))).predict({})
    assert info.value.status_code == 422 and info.value.details[0]["field"] == "age"
    assert APIClient("http://x", session=FakeSession(FakeResponse(204))).delete_customer("A1") is None
    with pytest.raises(APIError, match="Cannot reach"):
        APIClient("http://x", session=FakeSession(error=requests.ConnectionError())).health()
    with pytest.raises(APIError, match="HTTP 502"):
        APIClient("http://x", session=FakeSession(FakeResponse(502))).health()


@pytest.mark.parametrize(("raw", "expected"), [
    ("localhost:8000", "http://localhost:8000"), ("https://api.example.com/", "https://api.example.com"),
    ("credit-risk-api:10000", "http://credit-risk-api:10000"),
])
def test_base_url_normalisation(raw, expected):
    assert normalize_base_url(raw) == expected


def test_all_required_charts_build():
    fake = FakeClient()
    data, perf = fake.dashboard(), fake.model_performance()
    figures = [
        charts.risk_distribution_chart(data["risk_distribution"]),
        charts.histogram_chart(HIST, "Income", "income"),
        charts.histogram_chart({"bin_edges": [], "counts": []}, "Empty", "x"),
        charts.trend_chart(data["prediction_trend"]),
        charts.feature_importance_chart(perf["feature_importance"]),
        charts.curve_chart(perf["roc_curve"], "ROC", "FPR", "TPR", baseline=True),
        charts.curve_chart(perf["pr_curve"], "PR", "Recall", "Precision"),
        charts.confusion_matrix_chart(perf["confusion_matrix"]),
        charts.probability_gauge(0.2, 0.10, 0.30),
    ]
    assert all(f.to_dict()["data"] is not None for f in figures)
    assert charts.risk_distribution_chart(data["risk_distribution"]).data[0].values == (5, 2, 1)
