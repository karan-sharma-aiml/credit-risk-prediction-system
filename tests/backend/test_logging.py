import json
import logging

from backend.app.core.logging import JsonFormatter, pseudonymize, request_id_ctx
from tests.backend.payloads import GOOD_CUSTOMER


class Collector(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(JsonFormatter().format(record))


def test_logs_are_json_with_request_id_and_never_contain_customer_data(client):
    collector = Collector()
    logging.getLogger().addHandler(collector)
    logging.getLogger().setLevel(logging.DEBUG)
    try:
        created = client.post("/predict", json={**GOOD_CUSTOMER, "customer_id": "CUST_LOG_1"}).json()
        client.get("/customers/CUST_LOG_1")
        client.post("/predict", json={**GOOD_CUSTOMER, "credit_score": 5000})  # validation failure
    finally:
        logging.getLogger().removeHandler(collector)

    records = [json.loads(line) for line in collector.lines]
    messages = {r["message"] for r in records}
    assert {"request_completed", "prediction_completed", "validation_failed"} <= messages
    assert all({"ts", "level", "logger", "message"} <= set(r) for r in records)
    assert any(r.get("request_id") == created["request_id"] for r in records)

    blob = "\n".join(collector.lines)
    for secret in ("CUST_LOG_1", "95000", "780", "60000", str(created["default_probability"])):
        assert secret not in blob, f"{secret} leaked into logs"
    done = next(r for r in records if r["message"] == "prediction_completed")
    assert len(done["customer_ref"]) == 12 and "latency_ms" in done


def test_formatter_redacts_sensitive_extra_fields_and_logs_exceptions():
    logger = logging.getLogger("redaction-test")
    record = logger.makeRecord("redaction-test", logging.ERROR, __file__, 1, "msg", (), None,
                               extra={"credit_score": 700, "customer_id": "X1", "safe": "ok"})
    data = json.loads(JsonFormatter().format(record))
    assert data["credit_score"] == "[REDACTED]" and data["customer_id"] == "[REDACTED]" and data["safe"] == "ok"
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        rec = logger.makeRecord("t", logging.ERROR, __file__, 1, "failed", (), sys.exc_info())
    assert json.loads(JsonFormatter().format(rec))["exc_type"] == "ValueError"


def test_pseudonymize_is_stable_keyed_and_irreversible_looking():
    assert pseudonymize("CUST1", "k1") == pseudonymize("CUST1", "k1")
    assert pseudonymize("CUST1", "k1") != pseudonymize("CUST1", "k2") != pseudonymize("CUST2", "k2")
    assert "CUST1" not in pseudonymize("CUST1", "k1")


def test_request_id_context_is_reset_after_request(client):
    client.get("/health")
    assert request_id_ctx.get() is None
