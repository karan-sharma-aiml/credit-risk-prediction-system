"""Model registry and metrics persistence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from backend.app.database.models import ModelMetric, ModelRegistry


class ModelRegistryRepository:
    """Register model versions and store their evaluation metrics."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_version(self, model_name: str, model_version: str) -> ModelRegistry | None:
        stmt = select(ModelRegistry).where(
            ModelRegistry.model_name == model_name, ModelRegistry.model_version == model_version
        )
        return self.session.scalars(stmt).first()

    def register(self, **fields) -> ModelRegistry:
        model = ModelRegistry(**fields)
        self.session.add(model)
        self.session.flush()
        return model

    def activate(self, model: ModelRegistry) -> None:
        """Make ``model`` the single active version of its name."""
        self.session.execute(
            update(ModelRegistry)
            .where(ModelRegistry.model_name == model.model_name, ModelRegistry.model_id != model.model_id)
            .values(is_active=False)
        )
        model.is_active = True
        self.session.flush()

    def upsert_metrics(self, model_id: int, metrics: Mapping[tuple[str, str], float]) -> int:
        """Insert or update ``{(split, metric_name): value}``; returns the number of rows touched."""
        existing = {
            (m.dataset_split, m.metric_name): m
            for m in self.session.scalars(select(ModelMetric).where(ModelMetric.model_id == model_id))
        }
        for (split, name), value in metrics.items():
            if (split, name) in existing:
                existing[(split, name)].metric_value = float(value)
            else:
                self.session.add(
                    ModelMetric(model_id=model_id, dataset_split=split, metric_name=name, metric_value=float(value))
                )
        self.session.flush()
        return len(metrics)

    def list_metrics(self, model_id: int) -> Sequence[ModelMetric]:
        stmt = select(ModelMetric).where(ModelMetric.model_id == model_id).order_by(
            ModelMetric.dataset_split, ModelMetric.metric_name
        )
        return self.session.scalars(stmt).all()
