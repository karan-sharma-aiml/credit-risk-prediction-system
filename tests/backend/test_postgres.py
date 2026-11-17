"""Runs against a real PostgreSQL when TEST_DATABASE_URL is set (CI service container / local).

    TEST_DATABASE_URL=postgresql://user:pass@localhost:5432/testdb pytest tests/backend/test_postgres.py
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text

from backend.app.core.config import Settings
from backend.app.database.base import DB_SCHEMA, Base
from backend.app.main import create_app
from ml.config import PROJECT_ROOT
from tests.backend.payloads import GOOD_CUSTOMER, RISKY_CUSTOMER

URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")


def _reset_schema(url: str) -> None:
    engine = create_engine(Settings(database_url=url).sqlalchemy_url)
    with engine.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {DB_SCHEMA} CASCADE"))
    engine.dispose()


@pytest.fixture
def pg_client(predictor):
    _reset_schema(URL)
    settings = Settings(database_url=URL, environment="test", log_level="WARNING", secret_key="t")
    with TestClient(create_app(settings, predictor=predictor)) as client:
        yield client


def test_full_flow_on_postgres(pg_client):
    c = pg_client
    assert c.get("/health").json()["database"] == "up"
    good = c.post("/predict", json={**GOOD_CUSTOMER, "customer_id": "PG_1"}).json()
    risky = c.post("/predict", json={**RISKY_CUSTOMER, "customer_id": "PG_1"}).json()
    detail = c.get(f"/predictions/{risky['prediction_id']}").json()
    assert detail["history"]["previous_prediction_id"] == good["prediction_id"]
    assert detail["history"]["input_snapshot"]["credit_score"] == 420
    assert c.get("/customers").json()["total"] == 1
    dash = c.get("/dashboard").json()
    assert dash["kpis"]["total_predictions"] == 2 and sum(b["count"] for b in dash["risk_distribution"]) == 1
    assert dash["prediction_trend"][0]["count"] == 2
    assert dash["prediction_trend"][0]["date"] == datetime.now(timezone.utc).date().isoformat()  # not DB-local day
    assert c.delete("/customers/PG_1").status_code == 204
    assert c.get(f"/predictions/{good['prediction_id']}").status_code == 404


def test_check_constraints_and_pool_on_postgres(pg_client):
    engine = pg_client.app.state.session_factory.kw["bind"]
    assert engine.dialect.name == "postgresql" and engine.pool.size() == 5
    response = pg_client.post("/customers", json={**GOOD_CUSTOMER, "customer_id": "PG_2"})
    assert response.status_code == 201


def test_sql_files_create_the_same_tables_as_the_orm():
    """Apply sql/schema/*.sql to a clean schema and compare with the ORM metadata."""
    _reset_schema(URL)
    engine = create_engine(Settings(database_url=URL).sqlalchemy_url)
    raw = engine.raw_connection()
    try:
        cursor = raw.cursor()
        for path in sorted(Path(PROJECT_ROOT, "sql", "schema").glob("*.sql")):
            cursor.execute(path.read_text())
        raw.commit()
    finally:
        raw.close()
    inspector = inspect(engine)
    sql_tables = {t: {c["name"] for c in inspector.get_columns(t, schema=DB_SCHEMA)}
                  for t in inspector.get_table_names(schema=DB_SCHEMA)}
    for table in Base.metadata.tables.values():
        assert {c.name for c in table.columns} == sql_tables[table.name], table.name
    engine.dispose()
