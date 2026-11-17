"""Scoring endpoint."""

from __future__ import annotations

from fastapi import APIRouter, status

from backend.app.core.logging import request_id_ctx
from backend.app.dependencies import PredictionServiceDep
from backend.app.schemas.prediction import PredictionDetail, PredictionRequest

router = APIRouter(tags=["predictions"])


@router.post(
    "/predict",
    response_model=PredictionDetail,
    status_code=status.HTTP_201_CREATED,
    summary="Score a customer and store the prediction",
)
def predict(payload: PredictionRequest, service: PredictionServiceDep) -> PredictionDetail:
    """Validate -> score with the saved model -> persist customer, prediction and audit row."""
    return service.predict(payload, request_id=request_id_ctx.get())
