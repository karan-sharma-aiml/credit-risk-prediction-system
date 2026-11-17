"""Schemas for analytics and model information endpoints."""

from __future__ import annotations

from pydantic import BaseModel

from backend.app.schemas.prediction import PredictionRead


class KPIs(BaseModel):
    total_customers: int
    total_predictions: int
    average_default_probability: float | None
    high_risk_share: float | None


class RiskBucket(BaseModel):
    segment: str
    label: str
    count: int
    share: float


class Histogram(BaseModel):
    bin_edges: list[float]
    counts: list[int]


class TrendPoint(BaseModel):
    date: str
    count: int
    avg_probability: float


class DashboardSummary(BaseModel):
    """Response of ``GET /dashboard``."""

    kpis: KPIs
    risk_distribution: list[RiskBucket]
    income_distribution: Histogram
    loan_distribution: Histogram
    credit_score_distribution: Histogram
    prediction_trend: list[TrendPoint]
    recent_predictions: list[PredictionRead]


class ModelInfo(BaseModel):
    """Response of ``GET /model-info``."""

    model_name: str
    model_version: str
    decision_threshold: float
    trained_at: str | None = None
    dataset_version: str | None = None
    balanced_with_smote: bool | None = None
    random_state: int | None = None
    segment_cutoffs: dict[str, float]
    segment_labels: dict[str, str]
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    n_features: int


class CurvePoints(BaseModel):
    x: list[float]
    y: list[float]
    auc: float


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class ModelPerformance(BaseModel):
    """Response of ``GET /model-info/performance`` (test-set diagnostics)."""

    roc_curve: CurvePoints | None = None
    pr_curve: CurvePoints | None = None
    confusion_matrix: dict[str, int] | None = None
    feature_importance: list[FeatureImportance] = []
