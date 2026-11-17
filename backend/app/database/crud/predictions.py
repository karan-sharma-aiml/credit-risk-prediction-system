"""Prediction and prediction-history persistence."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from backend.app.database.models import Prediction, PredictionHistory, utcnow


class PredictionRepository:
    """CRUD and aggregate queries for :class:`Prediction`."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, prediction: Prediction, history: PredictionHistory) -> Prediction:
        """Insert a prediction and its audit row in the same transaction."""
        prediction.history = history
        self.session.add(prediction)
        self.session.flush()
        return prediction

    def get(self, prediction_id: int) -> Prediction | None:
        stmt = (
            select(Prediction)
            .options(joinedload(Prediction.model), joinedload(Prediction.history))
            .where(Prediction.prediction_id == prediction_id)
        )
        return self.session.scalars(stmt).unique().one_or_none()

    def _filtered(
        self,
        customer_id: str | None,
        risk_segment: str | None,
        date_from: datetime | None,
        date_to: datetime | None,
    ):
        stmt = select(Prediction)
        if customer_id:
            stmt = stmt.where(Prediction.customer_id == customer_id)
        if risk_segment:
            stmt = stmt.where(Prediction.risk_segment == risk_segment)
        if date_from:
            stmt = stmt.where(Prediction.predicted_at >= date_from)
        if date_to:
            stmt = stmt.where(Prediction.predicted_at <= date_to)
        return stmt

    def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        customer_id: str | None = None,
        risk_segment: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> Sequence[Prediction]:
        stmt = (
            self._filtered(customer_id, risk_segment, date_from, date_to)
            .options(joinedload(Prediction.model))
            .order_by(Prediction.predicted_at.desc(), Prediction.prediction_id.desc())
            .limit(limit)
            .offset(offset)
        )
        return self.session.scalars(stmt).unique().all()

    def count(
        self,
        *,
        customer_id: str | None = None,
        risk_segment: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> int:
        stmt = select(func.count()).select_from(self._filtered(customer_id, risk_segment, date_from, date_to).subquery())
        return int(self.session.scalar(stmt) or 0)

    def latest_for_customer(self, customer_id: str) -> Prediction | None:
        stmt = (
            select(Prediction)
            .where(Prediction.customer_id == customer_id)
            .order_by(Prediction.predicted_at.desc(), Prediction.prediction_id.desc())
            .limit(1)
        )
        return self.session.scalars(stmt).first()

    # ------------------------------------------------------------------ aggregates
    def segment_counts(self) -> dict[str, int]:
        rows = self.session.execute(
            select(Prediction.risk_segment, func.count()).group_by(Prediction.risk_segment)
        ).all()
        return {segment: int(count) for segment, count in rows}

    def latest_segment_counts(self) -> dict[str, int]:
        """Segment of each customer's most recent prediction (current portfolio risk)."""
        latest_ids = select(func.max(Prediction.prediction_id)).group_by(Prediction.customer_id)
        rows = self.session.execute(
            select(Prediction.risk_segment, func.count())
            .where(Prediction.prediction_id.in_(latest_ids))
            .group_by(Prediction.risk_segment)
        ).all()
        return {segment: int(count) for segment, count in rows}

    def average_probability(self) -> float | None:
        value = self.session.scalar(select(func.avg(Prediction.default_probability)))
        return None if value is None else float(value)

    def daily_trend(self, days: int = 30) -> list[dict]:
        """Predictions per calendar day (UTC) with mean probability, oldest first."""
        since = utcnow() - timedelta(days=days)
        moment = Prediction.predicted_at
        if self.session.get_bind().dialect.name == "postgresql":
            moment = func.timezone("UTC", moment)  # group by UTC day, not the DB session's time zone
        day = func.date(moment)
        rows = self.session.execute(
            select(day, func.count(), func.avg(Prediction.default_probability))
            .where(Prediction.predicted_at >= since)
            .group_by(day)
            .order_by(day)
        ).all()
        return [{"date": str(d), "count": int(c), "avg_probability": float(a)} for d, c, a in rows]
