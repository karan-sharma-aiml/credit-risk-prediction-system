"""Prediction use-case: validate -> score with ``CreditRiskPredictor`` -> persist -> respond."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime

from sqlalchemy.orm import Session

from backend.app.core.exceptions import NotFoundError, PredictionError
from backend.app.database.crud import CustomerRepository, PredictionRepository
from backend.app.database.models import Prediction, PredictionHistory, utcnow
from backend.app.schemas.common import Page
from backend.app.schemas.prediction import (
    PredictionDetail,
    PredictionHistoryRead,
    PredictionRead,
    PredictionRequest,
    RecommendationOut,
)
from backend.app.services.customer_service import CustomerService
from backend.app.services.model_service import ModelContext
from backend.app.services.recommendation import RecommendationEngine, derive_features
from backend.app.utils.ids import ensure_utc

logger = logging.getLogger(__name__)


class PredictionService:
    """Score customers and manage prediction history."""

    def __init__(
        self,
        session: Session,
        context: ModelContext,
        recommender: RecommendationEngine,
        pseudonymize: Callable[[str], str],
    ) -> None:
        self.session = session
        self.context = context
        self.recommender = recommender
        self._pseudo = pseudonymize
        self.customers = CustomerService(session, pseudonymize)
        self.predictions = PredictionRepository(session)
        self.customer_repo = CustomerRepository(session)
        seg = context.predictor.bundle.segment_settings
        self._labels = dict(zip(seg.codes, seg.labels))

    # ------------------------------------------------------------------ scoring
    def predict(
        self,
        request: PredictionRequest,
        *,
        request_id: str | None = None,
        created_at: datetime | None = None,
    ) -> PredictionDetail:
        """Score a customer and save customer + prediction + audit row atomically."""
        started = time.perf_counter()
        features = request.model_dump(exclude={"customer_id"})
        customer = self.customers.upsert_features(request.customer_id, features)
        record = {**features, "customer_id": customer.customer_id}

        try:
            result = self.context.predictor.predict_one(record)
        except Exception as exc:
            raise PredictionError("Model inference failed.") from exc

        recommendation = self.recommender.recommend(result["risk_segment"], derive_features(features))
        previous = self.predictions.latest_for_customer(customer.customer_id)
        when = created_at or utcnow()
        probability = float(result["default_probability"])

        prediction = Prediction(
            customer_id=customer.customer_id,
            model_id=self.context.model_id,
            default_probability=probability,
            predicted_label=int(result["predicted_label"]),
            decision_threshold=self.context.predictor.bundle.decision_threshold,
            risk_segment=result["risk_segment"],
            recommendation_action=recommendation.action,
            top_risk_factors=recommendation.considerations,
            request_id=request_id,
            predicted_at=when,
        )
        history = PredictionHistory(
            customer_id=customer.customer_id,
            input_snapshot=features,
            previous_prediction_id=previous.prediction_id if previous else None,
            previous_probability=previous.default_probability if previous else None,
            previous_segment=previous.risk_segment if previous else None,
            segment_changed=bool(previous and previous.risk_segment != result["risk_segment"]),
            probability_change=probability - previous.default_probability if previous else None,
            recorded_at=when,
        )
        self.predictions.create(prediction, history)
        self.session.commit()

        logger.info(
            "prediction_completed",
            extra={
                "prediction_id": prediction.prediction_id,
                "customer_ref": self._pseudo(customer.customer_id),
                "risk_segment": prediction.risk_segment,
                "model_version": result["model_version"],
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return self._detail(prediction, result["model_version"])

    # ------------------------------------------------------------------ queries
    def get(self, prediction_id: int) -> PredictionDetail:
        prediction = self.predictions.get(prediction_id)
        if prediction is None:
            raise NotFoundError(f"Prediction {prediction_id} not found.")
        return self._detail(prediction, prediction.model.model_version)

    def list(
        self,
        *,
        limit: int,
        offset: int,
        customer_id: str | None = None,
        risk_segment: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> Page[PredictionRead]:
        filters = {
            "customer_id": customer_id,
            "risk_segment": risk_segment,
            "date_from": date_from,
            "date_to": date_to,
        }
        rows = self.predictions.list(limit=limit, offset=offset, **filters)
        return Page[PredictionRead](
            items=[self.to_read(p) for p in rows],
            total=self.predictions.count(**filters),
            limit=limit,
            offset=offset,
        )

    # ------------------------------------------------------------------ mapping
    def to_read(self, prediction: Prediction, model_version: str | None = None) -> PredictionRead:
        """Map an ORM prediction to its API schema."""
        recommendation = self.recommender.from_stored(
            prediction.recommendation_action or "", prediction.top_risk_factors
        )
        return PredictionRead(
            prediction_id=prediction.prediction_id,
            customer_id=prediction.customer_id,
            default_probability=float(prediction.default_probability),
            predicted_label=int(prediction.predicted_label),
            risk_segment=prediction.risk_segment,
            risk_label=self._labels.get(prediction.risk_segment, prediction.risk_segment),
            recommendation=RecommendationOut(**recommendation.__dict__),
            model_version=model_version or prediction.model.model_version,
            decision_threshold=prediction.decision_threshold,
            created_at=ensure_utc(prediction.predicted_at),
        )

    def _detail(self, prediction: Prediction, model_version: str) -> PredictionDetail:
        base = self.to_read(prediction, model_version)
        history = prediction.history
        return PredictionDetail(
            **base.model_dump(),
            request_id=prediction.request_id,
            history=None
            if history is None
            else PredictionHistoryRead(
                input_snapshot=history.input_snapshot,
                previous_prediction_id=history.previous_prediction_id,
                previous_probability=history.previous_probability,
                previous_segment=history.previous_segment,
                segment_changed=history.segment_changed,
                probability_change=history.probability_change,
                recorded_at=ensure_utc(history.recorded_at),
            ),
        )
