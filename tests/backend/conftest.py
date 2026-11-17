"""Fixtures for API / database integration tests (real model artifacts, in-memory SQLite)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app
from ml.config import PATHS
from ml.predict import CreditRiskPredictor


@pytest.fixture(scope="session")
def predictor() -> CreditRiskPredictor:
    """The shipped champion model (never retrained by tests)."""
    if not PATHS.model_bundle.is_file():
        pytest.skip("saved_models/model_bundle.joblib not present")
    return CreditRiskPredictor.from_path(PATHS.model_bundle)


@pytest.fixture
def settings() -> Settings:
    return Settings(database_url="sqlite://", environment="test", log_level="WARNING", secret_key="test-secret")


@pytest.fixture
def client(settings: Settings, predictor: CreditRiskPredictor):
    """Fresh app + empty in-memory database per test."""
    with TestClient(create_app(settings, predictor=predictor)) as test_client:
        yield test_client


@pytest.fixture
def session_factory(client: TestClient):
    return client.app.state.session_factory
