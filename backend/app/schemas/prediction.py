"""Prediction request/response schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from backend.app.schemas.customer import CUSTOMER_ID_PATTERN, CustomerBase


class PredictionRequest(CustomerBase):
    """Body of ``POST /predict``.

    If ``customer_id`` is omitted a new customer is created; if it exists, the
    stored customer is updated with the submitted data before scoring.
    """

    customer_id: str | None = Field(default=None, pattern=CUSTOMER_ID_PATTERN)


class RecommendationOut(BaseModel):
    """Business recommendation derived from the risk segment."""

    action: str
    title: str
    message: str
    considerations: list[str] = []


class PredictionRead(BaseModel):
    """A stored prediction."""

    model_config = ConfigDict(from_attributes=True)

    prediction_id: int
    customer_id: str
    default_probability: float
    predicted_label: int
    risk_segment: str
    risk_label: str
    recommendation: RecommendationOut
    model_version: str
    decision_threshold: float | None = None
    created_at: datetime


class PredictionHistoryRead(BaseModel):
    """Audit information stored with a prediction."""

    input_snapshot: dict
    previous_prediction_id: int | None = None
    previous_probability: float | None = None
    previous_segment: str | None = None
    segment_changed: bool = False
    probability_change: float | None = None
    recorded_at: datetime


class PredictionDetail(PredictionRead):
    """Prediction plus its audit trail (``GET /predictions/{id}``)."""

    request_id: str | None = None
    history: PredictionHistoryRead | None = None
