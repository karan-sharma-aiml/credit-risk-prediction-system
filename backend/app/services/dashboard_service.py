"""Aggregations that power ``GET /dashboard``."""

from __future__ import annotations

import numpy as np
from sqlalchemy.orm import Session

from backend.app.database.crud import CustomerRepository, PredictionRepository
from backend.app.schemas.dashboard import DashboardSummary, Histogram, KPIs, RiskBucket, TrendPoint
from backend.app.services.model_service import ModelContext
from backend.app.services.prediction_service import PredictionService

HISTOGRAM_BINS = 20
TREND_DAYS = 30
RECENT_LIMIT = 10


def build_histogram(values: list[float], bins: int = HISTOGRAM_BINS) -> Histogram:
    """Equal-width histogram (empty when there is no data)."""
    if not values:
        return Histogram(bin_edges=[], counts=[])
    counts, edges = np.histogram(np.asarray(values, dtype=float), bins=bins)
    return Histogram(bin_edges=np.round(edges, 2).tolist(), counts=counts.astype(int).tolist())


class DashboardService:
    """Portfolio KPIs, distributions and trends."""

    def __init__(self, session: Session, prediction_service: PredictionService, context: ModelContext) -> None:
        self.customers = CustomerRepository(session)
        self.predictions = PredictionRepository(session)
        self.prediction_service = prediction_service
        seg = context.predictor.bundle.segment_settings
        self._segments = list(zip(seg.codes, seg.labels))

    def summary(self) -> DashboardSummary:
        counts = self.predictions.latest_segment_counts()
        scored_customers = sum(counts.values())
        total_predictions = self.predictions.count()
        buckets = [
            RiskBucket(
                segment=code,
                label=label,
                count=counts.get(code, 0),
                share=round(counts.get(code, 0) / scored_customers, 4) if scored_customers else 0.0,
            )
            for code, label in self._segments
        ]
        high_share = next((b.share for b in buckets if b.segment == "high"), None) if scored_customers else None
        recent = self.predictions.list(limit=RECENT_LIMIT)
        return DashboardSummary(
            kpis=KPIs(
                total_customers=self.customers.count(),
                total_predictions=total_predictions,
                average_default_probability=self.predictions.average_probability(),
                high_risk_share=high_share,
            ),
            risk_distribution=buckets,
            income_distribution=build_histogram(self.customers.numeric_values("annual_income")),
            loan_distribution=build_histogram(self.customers.numeric_values("loan_amount")),
            credit_score_distribution=build_histogram(self.customers.numeric_values("credit_score")),
            prediction_trend=[TrendPoint(**row) for row in self.predictions.daily_trend(TREND_DAYS)],
            recent_predictions=[self.prediction_service.to_read(p) for p in recent],
        )
