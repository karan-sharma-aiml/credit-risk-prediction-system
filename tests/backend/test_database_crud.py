"""Repository (CRUD layer) tests against the ORM, independent of the HTTP layer."""

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from backend.app.database.crud import CustomerRepository, ModelRegistryRepository, PredictionRepository
from backend.app.database.models import Customer, ModelMetric, Prediction, PredictionHistory
from tests.backend.payloads import GOOD_CUSTOMER


def _customer(repo, cid="CUST_DB1", **overrides):
    return repo.create({"customer_id": cid, **GOOD_CUSTOMER, **overrides})


def _prediction(model_id, cid, probability=0.2, segment="medium"):
    return (
        Prediction(customer_id=cid, model_id=model_id, default_probability=probability, predicted_label=1,
                   decision_threshold=0.09, risk_segment=segment, recommendation_action="manual_review"),
        PredictionHistory(customer_id=cid, input_snapshot={"credit_score": 700}),
    )


@pytest.fixture
def session(session_factory):
    with session_factory() as s:
        yield s


@pytest.fixture
def model_id(client):
    return client.app.state.model_context.model_id


def test_customer_create_get_update_delete(session):
    repo = CustomerRepository(session)
    created = _customer(repo)
    session.commit()
    assert repo.get("CUST_DB1").credit_score == 780 and created.created_at is not None
    repo.update(created, {"credit_score": 650})
    session.commit()
    assert repo.get("CUST_DB1").credit_score == 650
    repo.delete(created)
    session.commit()
    assert repo.get("CUST_DB1") is None


def test_customer_list_count_search_and_numeric_values(session):
    repo = CustomerRepository(session)
    for i in range(3):
        _customer(repo, f"AAA{i}", annual_income=50_000 + i)
    _customer(repo, "BBB0")
    session.commit()
    assert repo.count() == 4 and repo.count("AAA") == 3
    assert len(repo.list(limit=2, offset=0)) == 2 and len(repo.list(limit=10, offset=3)) == 1
    assert sorted(repo.numeric_values("annual_income"))[:3] == [50_000, 50_001, 50_002]


def test_database_check_constraints_are_enforced(session):
    repo = CustomerRepository(session)
    with pytest.raises(IntegrityError):
        _customer(repo, "BAD1", age=5)
    session.rollback()
    with pytest.raises(IntegrityError):
        _customer(repo, "BAD2", employment_type="astronaut")


def test_integrity_error_message_hides_bound_values(session):
    repo = CustomerRepository(session)
    with pytest.raises(IntegrityError) as info:
        _customer(repo, "BAD3", age=5, annual_income=123456.78)
    assert "123456.78" not in str(info.value)  # engine created with hide_parameters=True


def test_prediction_create_get_aggregates(session, model_id):
    customers, predictions = CustomerRepository(session), PredictionRepository(session)
    _customer(customers, "C1")
    _customer(customers, "C2")
    for cid, p, seg in [("C1", 0.05, "low"), ("C1", 0.4, "high"), ("C2", 0.15, "medium")]:
        predictions.create(*_prediction(model_id, cid, p, seg))
    session.commit()

    assert predictions.count() == 3 and predictions.count(customer_id="C1") == 2
    assert predictions.segment_counts() == {"low": 1, "high": 1, "medium": 1}
    assert predictions.latest_segment_counts() == {"high": 1, "medium": 1}  # latest per customer only
    assert predictions.latest_for_customer("C1").risk_segment == "high"
    assert predictions.average_probability() == pytest.approx(0.2)
    first = predictions.list(limit=1)[0]
    loaded = predictions.get(first.prediction_id)
    assert loaded.history.input_snapshot == {"credit_score": 700} and loaded.model.model_version
    assert predictions.daily_trend(7)[0]["count"] == 3


def test_foreign_key_and_cascade_behaviour(session, model_id):
    customers, predictions = CustomerRepository(session), PredictionRepository(session)
    c = _customer(customers, "C9")
    predictions.create(*_prediction(model_id, "C9"))
    session.commit()
    customers.delete(c)
    session.commit()
    assert session.scalar(select(Prediction).where(Prediction.customer_id == "C9")) is None
    assert session.scalar(select(PredictionHistory).where(PredictionHistory.customer_id == "C9")) is None
    with pytest.raises(IntegrityError):  # prediction for a customer that does not exist
        predictions.create(*_prediction(model_id, "NOPE"))


def test_model_registry_single_active_and_idempotent_metrics(session):
    repo = ModelRegistryRepository(session)
    a = repo.register(model_name="m", model_version="v1")
    b = repo.register(model_name="m", model_version="v2")
    repo.activate(a)
    repo.activate(b)
    session.commit()
    assert (a.is_active, b.is_active) == (False, True)
    assert repo.get_by_version("m", "v2").model_id == b.model_id

    repo.upsert_metrics(b.model_id, {("test", "recall"): 0.7, ("test", "f1"): 0.4})
    repo.upsert_metrics(b.model_id, {("test", "recall"): 0.8})
    session.commit()
    metrics = {(m.dataset_split, m.metric_name): m.metric_value for m in repo.list_metrics(b.model_id)}
    assert metrics == {("test", "recall"): 0.8, ("test", "f1"): 0.4}
    with pytest.raises(IntegrityError):
        session.add(ModelMetric(model_id=b.model_id, dataset_split="oops", metric_name="x", metric_value=1))
        session.flush()


def test_startup_registers_model_and_stores_metrics(client, session):
    repo = ModelRegistryRepository(session)
    model = repo.get_by_version(*(client.get("/model-info").json()[k] for k in ("model_name", "model_version")))
    assert model is not None and model.is_active and model.algorithm == "RandomForestClassifier"
    names = {(m.dataset_split, m.metric_name) for m in repo.list_metrics(model.model_id)}
    assert {("validation", "recall"), ("test", "roc_auc"), ("test", "tp")} <= names
    assert session.scalar(select(Customer)) is None
