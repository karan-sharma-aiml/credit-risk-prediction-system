"""Prediction history endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Query

from backend.app.dependencies import PredictionServiceDep
from backend.app.schemas.common import Page
from backend.app.schemas.customer import CUSTOMER_ID_PATTERN
from backend.app.schemas.prediction import PredictionDetail, PredictionRead

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("", response_model=Page[PredictionRead], summary="Prediction history (newest first, filterable)")
def list_predictions(
    service: PredictionServiceDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    customer_id: Annotated[str | None, Query(pattern=CUSTOMER_ID_PATTERN)] = None,
    risk_segment: Literal["low", "medium", "high"] | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> Page[PredictionRead]:
    return service.list(
        limit=limit, offset=offset, customer_id=customer_id, risk_segment=risk_segment,
        date_from=date_from, date_to=date_to,
    )


@router.get("/{prediction_id}", response_model=PredictionDetail, summary="One prediction with its audit trail")
def get_prediction(prediction_id: int, service: PredictionServiceDep) -> PredictionDetail:
    return service.get(prediction_id)
