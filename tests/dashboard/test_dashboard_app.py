"""Dashboard startup and page smoke tests using Streamlit's AppTest."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from dashboard.api_client import APIError
from ml.config import PROJECT_ROOT
from tests.dashboard.conftest import FakeClient

APP = str(Path(PROJECT_ROOT, "dashboard", "app.py"))
PAGES = ["Dashboard", "Customer Management", "Prediction", "Analytics", "Prediction History"]


def _run(**_):
    return AppTest.from_file(APP, default_timeout=30).run()


def _open(page: str) -> AppTest:
    at = _run()
    at.sidebar.radio[0].set_value(page).run()
    return at


def test_dashboard_starts_and_renders_kpis(fake_client):
    at = _run()
    assert not at.exception
    assert [m.label for m in at.metric] == ["Customers", "Predictions", "Avg default probability", "High-risk customers"]
    assert at.sidebar.radio[0].options == PAGES
    assert any("API online" in s.value for s in at.sidebar.success)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_exception(fake_client, page):
    at = _open(page)
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value == page


def test_dashboard_empty_state_shows_guidance(monkeypatch):
    monkeypatch.setattr("dashboard.api_client.get_client", lambda: FakeClient(empty=True))
    at = _run()
    assert not at.exception and any("No predictions yet" in i.value for i in at.info)


def test_app_survives_unreachable_api(monkeypatch):
    class Down(FakeClient):
        def health(self):
            raise APIError("Cannot reach the API at http://x.")

    monkeypatch.setattr("dashboard.api_client.get_client", lambda: Down())
    at = _run()
    assert not at.exception
    assert any("Cannot reach the API" in e.value for e in at.error)


def test_prediction_form_submits_payload_and_shows_recommendation(fake_client):
    at = _open("Prediction")
    at.button[0].click().run()
    assert not at.exception
    name, payload = fake_client.calls[-1]
    assert name == "predict" and payload["credit_score"] == 680 and "customer_id" not in payload
    text = " ".join(m.value for m in at.markdown)
    assert "Reject Loan" in text and "High Risk" in text
    assert any("Debt-to-income" in w.value for w in at.warning)


def test_existing_customer_prediction_sends_customer_id(fake_client):
    at = _open("Prediction")
    at.radio(key="pred_mode").set_value("Existing customer").run()
    at.text_input(key="pred_customer_id").set_value("CUST_UI_1").run()
    at.button[0].click().run()
    name, payload = fake_client.calls[-1]
    assert name == "predict" and payload["customer_id"] == "CUST_UI_1" and payload["credit_score"] == 700


def test_unknown_existing_customer_shows_api_error(fake_client):
    at = _open("Prediction")
    at.radio(key="pred_mode").set_value("Existing customer").run()
    at.text_input(key="pred_customer_id").set_value("NOPE_1").run()
    assert any("not found" in e.value for e in at.error)


def test_customer_create_form_calls_api(fake_client):
    at = _open("Customer Management")
    submit = [b for b in at.button if b.label == "Create customer"][0]
    submit.click().run()
    assert not at.exception
    assert fake_client.calls[-1][0] == "create_customer"
    assert any("CUST_NEW" in s.value for s in at.success)


def test_history_page_applies_filters(fake_client):
    at = _open("Prediction History")
    at.selectbox(key="hist_segment").set_value("high").run()
    name, filters = fake_client.calls[-1]
    assert name == "list_predictions" and filters["risk_segment"] == "high"
