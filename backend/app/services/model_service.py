"""Model metadata: registration in the database, model info and test-set diagnostics.

Inference itself is NOT implemented here; it always goes through
``ml.predict.CreditRiskPredictor``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sqlalchemy.orm import Session

from backend.app.database.crud import ModelRegistryRepository
from backend.app.schemas.dashboard import CurvePoints, FeatureImportance, ModelInfo, ModelPerformance
from ml.predict import CreditRiskPredictor

logger = logging.getLogger(__name__)

MAX_CURVE_POINTS = 200


@dataclass(frozen=True)
class ModelContext:
    """The loaded predictor together with its ``model_registry`` primary key."""

    predictor: CreditRiskPredictor
    model_id: int


def _flatten_metrics(metrics: dict) -> dict[tuple[str, str], float]:
    """``{"validation": {...}, "test": {...}}`` -> ``{(split, name): value}`` incl. confusion cells."""
    flat: dict[tuple[str, str], float] = {}
    for split, values in metrics.items():
        for name, value in values.items():
            if name == "confusion_matrix":
                flat.update({(split, cell): float(count) for cell, count in value.items()})
            else:
                flat[(split, name)] = float(value)
    return flat


def _json_safe(params: dict) -> dict:
    return {k: v if isinstance(v, (int, float, str, bool, type(None))) else str(v) for k, v in params.items()}


def register_active_model(session: Session, predictor: CreditRiskPredictor, artifact_path: Path) -> int:
    """Register the loaded model (idempotent), mark it active and store its metrics.

    Returns:
        The ``model_registry.model_id`` to attach to predictions.
    """
    bundle = predictor.bundle
    repo = ModelRegistryRepository(session)
    model = repo.get_by_version(bundle.model_name, bundle.model_version)
    if model is None:
        classifier = bundle.pipeline.named_steps["model"]
        model = repo.register(
            model_name=bundle.model_name,
            model_version=bundle.model_version,
            algorithm=type(classifier).__name__,
            artifact_path=str(artifact_path),
            dataset_version=bundle.metadata.get("dataset_version"),
            hyperparameters=_json_safe(classifier.get_params()),
            trained_at=pd.Timestamp(bundle.metadata["trained_at"]).to_pydatetime()
            if bundle.metadata.get("trained_at")
            else None,
        )
        logger.info("model_registered", extra={"model_version": bundle.model_version})
    repo.activate(model)
    repo.upsert_metrics(model.model_id, _flatten_metrics(bundle.metadata.get("metrics", {})))
    session.commit()
    return model.model_id


class ModelService:
    """Read-only view of the loaded model (info + cached diagnostics)."""

    def __init__(self, predictor: CreditRiskPredictor, reports_dir: Path) -> None:
        self.predictor = predictor
        self.reports_dir = reports_dir

    def info(self) -> ModelInfo:
        bundle = self.predictor.bundle
        meta = bundle.metadata
        seg = bundle.segment_settings

        def scalars(split: str) -> dict[str, float]:
            return {k: float(v) for k, v in meta.get("metrics", {}).get(split, {}).items() if k != "confusion_matrix"}

        return ModelInfo(
            model_name=bundle.model_name,
            model_version=bundle.model_version,
            decision_threshold=bundle.decision_threshold,
            trained_at=meta.get("trained_at"),
            dataset_version=meta.get("dataset_version"),
            balanced_with_smote=meta.get("balanced_with_smote"),
            random_state=meta.get("random_state"),
            segment_cutoffs={"low_max": seg.low_max, "high_min": seg.high_min},
            segment_labels=dict(zip(seg.codes, seg.labels)),
            validation_metrics=scalars("validation"),
            test_metrics=scalars("test"),
            n_features=len(bundle.numeric_features) + len(bundle.categorical_features),
        )

    @cached_property
    def _performance(self) -> ModelPerformance:
        """Computed once: curves from the saved test predictions, importances from the classifier."""
        roc = pr = None
        path = self.reports_dir / "test_predictions.csv"
        if path.is_file():
            data = pd.read_csv(path)
            y, p = data["actual_default"].to_numpy(), data["default_probability"].to_numpy()
            roc, pr = self._roc(y, p), self._pr(y, p)
        else:
            logger.warning("test_predictions_missing", extra={"path": str(path)})
        confusion = self.predictor.bundle.metadata.get("metrics", {}).get("test", {}).get("confusion_matrix")
        return ModelPerformance(
            roc_curve=roc, pr_curve=pr, confusion_matrix=confusion, feature_importance=self._importances()
        )

    def performance(self) -> ModelPerformance:
        return self._performance

    @staticmethod
    def _thin(*arrays: np.ndarray) -> list[list[float]]:
        idx = np.unique(np.linspace(0, len(arrays[0]) - 1, min(len(arrays[0]), MAX_CURVE_POINTS)).astype(int))
        return [np.round(a[idx], 5).tolist() for a in arrays]

    def _roc(self, y: np.ndarray, p: np.ndarray) -> CurvePoints:
        fpr, tpr, _ = roc_curve(y, p)
        x, yy = self._thin(fpr, tpr)
        return CurvePoints(x=x, y=yy, auc=float(roc_auc_score(y, p)))

    def _pr(self, y: np.ndarray, p: np.ndarray) -> CurvePoints:
        precision, recall, _ = precision_recall_curve(y, p)
        x, yy = self._thin(recall, precision)
        return CurvePoints(x=x, y=yy, auc=float(average_precision_score(y, p)))

    def _importances(self, top: int = 15) -> list[FeatureImportance]:
        pipeline = self.predictor.bundle.pipeline
        model = pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            values = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            values = np.abs(np.ravel(model.coef_))
        else:
            return []
        names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        order = np.argsort(values)[::-1][:top]
        return [FeatureImportance(feature=str(names[i]), importance=round(float(values[i]), 5)) for i in order]
