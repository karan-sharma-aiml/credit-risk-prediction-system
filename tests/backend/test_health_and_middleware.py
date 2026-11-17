from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app
from tests.backend.payloads import GOOD_CUSTOMER


def test_health_ok(client):
    body = client.get("/health").json()
    assert client.get("/health").status_code == 200
    assert body["status"] == "ok" and body["database"] == "up" and body["model"].startswith("random_forest")


def test_health_degraded_when_model_missing(settings, tmp_path):
    broken = Settings(database_url="sqlite://", environment="test", log_level="WARNING", model_path=tmp_path / "nope.joblib")
    with TestClient(create_app(broken)) as c:
        response = c.get("/health")
        assert response.status_code == 503 and response.json()["model"] == "not_loaded"
        error = c.post("/predict", json=GOOD_CUSTOMER)
        assert error.status_code == 503 and error.json()["error"]["code"] == "model_unavailable"
        assert c.get("/model-info").status_code == 503


def test_health_degraded_when_database_down(client):
    def boom():
        raise RuntimeError("db down")

    client.app.state.session_factory = boom
    response = client.get("/health")
    assert response.status_code == 503 and response.json()["database"] == "down"


def test_request_id_header_generated_and_echoed(client):
    generated = client.get("/health").headers["X-Request-ID"]
    assert len(generated) >= 8
    assert client.get("/health", headers={"X-Request-ID": "abc-12345678"}).headers["X-Request-ID"] == "abc-12345678"
    assert client.get("/health", headers={"X-Request-ID": "bad id!"}).headers["X-Request-ID"] != "bad id!"


def test_unknown_route_uses_error_envelope(client):
    response = client.get("/nope")
    assert response.status_code == 404 and response.json()["error"]["code"] == "http_error"
