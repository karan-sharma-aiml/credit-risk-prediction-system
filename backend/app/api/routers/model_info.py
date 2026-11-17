"""Model metadata and test-set diagnostics."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.dependencies import ModelServiceDep
from backend.app.schemas.dashboard import ModelInfo, ModelPerformance

router = APIRouter(prefix="/model-info", tags=["model"])


@router.get("", response_model=ModelInfo, summary="Loaded model version, thresholds and metrics")
def model_info(service: ModelServiceDep) -> ModelInfo:
    return service.info()


@router.get("/performance", response_model=ModelPerformance, summary="ROC/PR curves, confusion matrix, feature importance")
def model_performance(service: ModelServiceDep) -> ModelPerformance:
    return service.performance()
