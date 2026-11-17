"""SQLAlchemy ORM models.

They mirror the Part 1 DDL in ``sql/schema`` (``tests/backend/test_schema_parity.py``
guards against drift). ``prediction_history`` is the append-only audit trail:
the exact input used for every prediction plus how the customer's risk moved
since the previous one.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database.base import DB_SCHEMA, Base

# BIGSERIAL on PostgreSQL, plain auto-increment INTEGER on SQLite.
BigIntPK = BigInteger().with_variant(Integer(), "sqlite")
Json = JSON().with_variant(JSONB(), "postgresql")


def _money() -> Numeric:
    return Numeric(14, 2, asdecimal=False)


def utcnow() -> datetime:
    """Timezone-aware current UTC time."""
    return datetime.now(timezone.utc)


class ModelRegistry(Base):
    """One row per trained model version that has served predictions."""

    __tablename__ = "model_registry"
    __table_args__ = (
        UniqueConstraint("model_name", "model_version"),
        Index(
            "uq_model_registry_one_active",
            "model_name",
            unique=True,
            postgresql_where=text("is_active"),
            sqlite_where=text("is_active"),
        ),
    )

    model_id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(100))
    model_version: Mapped[str] = mapped_column(String(50))
    algorithm: Mapped[str | None] = mapped_column(String(100))
    artifact_path: Mapped[str | None] = mapped_column(Text)
    dataset_version: Mapped[str | None] = mapped_column(String(50))
    hyperparameters: Mapped[dict | None] = mapped_column(Json)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, server_default=func.now())

    metrics: Mapped[list["ModelMetric"]] = relationship(back_populates="model", cascade="all, delete-orphan")


class Customer(Base):
    """Customer master data (the model's raw input schema + the observed outcome)."""

    __tablename__ = "customers"
    __table_args__ = (
        CheckConstraint("age BETWEEN 18 AND 100", name="age_range"),
        CheckConstraint("annual_income > 0", name="income_positive"),
        CheckConstraint(
            "employment_type IN ('salaried', 'self_employed', 'retired', 'unemployed')", name="employment_type"
        ),
        CheckConstraint("employment_years BETWEEN 0 AND 60", name="employment_years"),
        CheckConstraint("loan_amount > 0", name="loan_amount_positive"),
        CheckConstraint("loan_term_months BETWEEN 1 AND 480", name="loan_term"),
        CheckConstraint("credit_score BETWEEN 300 AND 850", name="credit_score"),
        CheckConstraint("monthly_debt_payment >= 0", name="debt_non_negative"),
        CheckConstraint("credit_limit >= 0", name="credit_limit_non_negative"),
        CheckConstraint("revolving_balance >= 0", name="revolving_non_negative"),
        CheckConstraint("savings_balance >= 0", name="savings_non_negative"),
        CheckConstraint("monthly_expenses >= 0", name="expenses_non_negative"),
        CheckConstraint("num_existing_loans >= 0", name="existing_loans_non_negative"),
        CheckConstraint("num_late_payments >= 0", name="late_payments_non_negative"),
        CheckConstraint("repayment_history IN ('excellent', 'good', 'fair', 'poor')", name="repayment_history"),
        CheckConstraint("home_ownership IN ('rent', 'mortgage', 'own', 'other')", name="home_ownership"),
        Index("idx_customers_is_default", "is_default"),
        Index("idx_customers_source", "source_dataset"),
    )

    customer_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    age: Mapped[int | None] = mapped_column(SmallInteger)
    annual_income: Mapped[float | None] = mapped_column(_money())
    employment_type: Mapped[str | None] = mapped_column(String(20))
    employment_years: Mapped[float | None] = mapped_column(Numeric(4, 1, asdecimal=False))
    loan_amount: Mapped[float | None] = mapped_column(_money())
    loan_term_months: Mapped[int | None] = mapped_column(SmallInteger)
    credit_score: Mapped[int | None] = mapped_column(SmallInteger)
    monthly_debt_payment: Mapped[float | None] = mapped_column(Numeric(12, 2, asdecimal=False))
    credit_limit: Mapped[float | None] = mapped_column(_money())
    revolving_balance: Mapped[float | None] = mapped_column(_money())
    savings_balance: Mapped[float | None] = mapped_column(_money())
    monthly_expenses: Mapped[float | None] = mapped_column(Numeric(12, 2, asdecimal=False))
    num_existing_loans: Mapped[int | None] = mapped_column(SmallInteger)
    num_late_payments: Mapped[int | None] = mapped_column(SmallInteger)
    repayment_history: Mapped[str | None] = mapped_column(String(10))
    home_ownership: Mapped[str | None] = mapped_column(String(10))
    # Observed outcome ("default" in the ML dataset; DEFAULT is a reserved SQL keyword).
    is_default: Mapped[bool | None] = mapped_column(Boolean)
    source_dataset: Mapped[str] = mapped_column(String(100), default="api")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, server_default=func.now()
    )

    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="customer", cascade="all, delete-orphan", order_by="Prediction.prediction_id.desc()"
    )


class Prediction(Base):
    """Outcome of one scoring request (append-only)."""

    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint("default_probability BETWEEN 0 AND 1", name="probability_range"),
        CheckConstraint("predicted_label IN (0, 1)", name="label_binary"),
        CheckConstraint("decision_threshold BETWEEN 0 AND 1", name="threshold_range"),
        CheckConstraint("risk_segment IN ('low', 'medium', 'high')", name="risk_segment"),
        Index("idx_predictions_customer_time", "customer_id", "predicted_at"),
        Index("idx_predictions_model", "model_id"),
        Index("idx_predictions_segment", "risk_segment"),
    )

    prediction_id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(
        String(20), ForeignKey(f"{DB_SCHEMA}.customers.customer_id", ondelete="CASCADE")
    )
    model_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(f"{DB_SCHEMA}.model_registry.model_id"))
    default_probability: Mapped[float] = mapped_column(Numeric(7, 6, asdecimal=False))
    predicted_label: Mapped[int] = mapped_column(SmallInteger)
    decision_threshold: Mapped[float | None] = mapped_column(Numeric(5, 4, asdecimal=False))
    risk_segment: Mapped[str] = mapped_column(String(10))
    recommendation_action: Mapped[str | None] = mapped_column(String(20))
    # Rule-based considerations produced by the recommendation engine.
    top_risk_factors: Mapped[list | None] = mapped_column(Json)
    request_id: Mapped[str | None] = mapped_column(String(64))
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, server_default=func.now())

    customer: Mapped[Customer] = relationship(back_populates="predictions")
    model: Mapped[ModelRegistry] = relationship()
    history: Mapped["PredictionHistory | None"] = relationship(
        back_populates="prediction", cascade="all, delete-orphan", uselist=False
    )


class PredictionHistory(Base):
    """Audit trail: exact inputs of a prediction and the change vs. the customer's previous one."""

    __tablename__ = "prediction_history"
    __table_args__ = (
        UniqueConstraint("prediction_id"),
        Index("idx_prediction_history_customer", "customer_id", "recorded_at"),
    )

    history_id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey(f"{DB_SCHEMA}.predictions.prediction_id", ondelete="CASCADE")
    )
    customer_id: Mapped[str] = mapped_column(
        String(20), ForeignKey(f"{DB_SCHEMA}.customers.customer_id", ondelete="CASCADE")
    )
    input_snapshot: Mapped[dict] = mapped_column(Json)
    previous_prediction_id: Mapped[int | None] = mapped_column(BigInteger)
    previous_probability: Mapped[float | None] = mapped_column(Numeric(7, 6, asdecimal=False))
    previous_segment: Mapped[str | None] = mapped_column(String(10))
    segment_changed: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    probability_change: Mapped[float | None] = mapped_column(Float)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, server_default=func.now())

    prediction: Mapped[Prediction] = relationship(back_populates="history")


class ModelMetric(Base):
    """Evaluation metric of a registered model (long format)."""

    __tablename__ = "model_metrics"
    __table_args__ = (
        UniqueConstraint("model_id", "dataset_split", "metric_name"),
        CheckConstraint("dataset_split IN ('train', 'validation', 'test')", name="dataset_split"),
        Index("idx_model_metrics_name", "metric_name"),
    )

    metric_id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey(f"{DB_SCHEMA}.model_registry.model_id", ondelete="CASCADE")
    )
    dataset_split: Mapped[str] = mapped_column(String(12))
    metric_name: Mapped[str] = mapped_column(String(50))
    metric_value: Mapped[float] = mapped_column(Float)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, server_default=func.now())

    model: Mapped[ModelRegistry] = relationship(back_populates="metrics")
