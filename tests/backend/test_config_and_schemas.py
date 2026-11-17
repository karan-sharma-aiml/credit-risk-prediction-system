import pytest
from pydantic import ValidationError

from backend.app.core.config import Settings
from backend.app.schemas.customer import (
    CustomerBase,
    CustomerRead,
    EmploymentType,
    HomeOwnership,
    RepaymentHistory,
)
from ml.config import CATEGORY_LEVELS, PROJECT_ROOT
from tests.backend.payloads import GOOD_CUSTOMER


def test_database_url_is_normalised_for_sqlalchemy():
    assert Settings(database_url="postgres://u:p@h:5432/d").sqlalchemy_url == "postgresql+psycopg2://u:p@h:5432/d"
    assert Settings(database_url="postgresql://u:p@h/d").sqlalchemy_url == "postgresql+psycopg2://u:p@h/d"
    assert Settings(database_url="sqlite://").is_sqlite


def test_production_requires_real_secret_key():
    with pytest.raises(ValidationError):
        Settings(environment="production", secret_key="change-me")
    assert Settings(environment="production", secret_key="a-real-secret").environment == "production"


def test_environment_variables_override_defaults(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("API_PORT", "9001")
    monkeypatch.setenv("CORS_ORIGINS", "http://a.example, http://b.example")
    monkeypatch.setenv("MODEL_PATH", "saved_models/custom.joblib")
    s = Settings()
    assert (s.log_level, s.api_port) == ("DEBUG", 9001)
    assert s.cors_origin_list == ["http://a.example", "http://b.example"]
    assert s.model_path == PROJECT_ROOT / "saved_models/custom.joblib"


def test_secret_key_is_not_exposed_in_repr():
    assert "super-secret" not in repr(Settings(secret_key="super-secret"))


def test_schema_literals_match_ml_config_levels():
    from typing import get_args

    assert set(get_args(EmploymentType)) == set(CATEGORY_LEVELS["employment_type"])
    assert set(get_args(RepaymentHistory)) == set(CATEGORY_LEVELS["repayment_history"])
    assert set(get_args(HomeOwnership)) == set(CATEGORY_LEVELS["home_ownership"])


def test_customer_base_fields_cover_every_model_input():
    from ml.config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

    assert set(NUMERIC_COLUMNS) | set(CATEGORICAL_COLUMNS) == set(CustomerBase.model_fields)


def test_read_schema_tolerates_rows_that_request_schema_would_reject():
    row = {"customer_id": "X1", "age": 150, "source_dataset": "s", "created_at": "2026-01-01T00:00:00",
           "updated_at": "2026-01-01T00:00:00", "credit_score": None}
    assert CustomerRead.model_validate(row).age == 150
    with pytest.raises(ValidationError):
        CustomerBase(**{**GOOD_CUSTOMER, "age": 150})
