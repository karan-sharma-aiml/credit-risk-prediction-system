"""Inference: raw customer data -> probability, label and risk segment.

This module never fits anything. It applies the *same* row-preserving cleaning
and feature engineering as training, then the stored pipeline, prior
correction, threshold and segment cut-offs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ml.artifacts import ModelBundle, load_bundle
from ml.config import ID_COL
from ml.features import engineer_features
from ml.preprocessing import clean_for_inference
from ml.segmentation import assign_risk_segment

logger = logging.getLogger(__name__)

RawInput = pd.DataFrame | Mapping | Sequence[Mapping]


class CreditRiskPredictor:
    """Score raw customer records with a saved :class:`ModelBundle`."""

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle

    @classmethod
    def from_path(cls, path: Path | str | None = None) -> "CreditRiskPredictor":
        """Load the bundle from disk (default: ``saved_models/model_bundle.joblib``)."""
        return cls(load_bundle(path))

    @property
    def model_version(self) -> str:
        """Identifier of the loaded model."""
        return self.bundle.model_version

    @staticmethod
    def _to_frame(data: RawInput) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if isinstance(data, Mapping):
            return pd.DataFrame([data])
        return pd.DataFrame(list(data))

    def predict_proba(self, data: RawInput) -> np.ndarray:
        """Prior-corrected default probabilities for raw records."""
        frame = engineer_features(clean_for_inference(self._to_frame(data)))
        features = frame[self.bundle.numeric_features + self.bundle.categorical_features]
        raw = self.bundle.pipeline.predict_proba(features)[:, 1]
        return self.bundle.prior_correction.apply(raw)

    def predict(self, data: RawInput) -> pd.DataFrame:
        """Score raw records.

        Args:
            data: DataFrame, single record (mapping) or list of records, using
                the raw schema (see ``docs/dataset.md``). Missing values are
                allowed; unseen categories are ignored safely.

        Returns:
            One row per input row with ``default_probability``,
            ``predicted_label`` (1 = predicted default), ``risk_segment``
            (low/medium/high) and ``risk_label``; ``customer_id`` is included
            when supplied.
        """
        frame = self._to_frame(data)
        proba = self.predict_proba(frame)
        out = pd.DataFrame(index=frame.index)
        if ID_COL in frame.columns:
            out[ID_COL] = frame[ID_COL].to_numpy()
        out["default_probability"] = np.round(proba, 6)
        out["predicted_label"] = (proba >= self.bundle.decision_threshold).astype(int)
        segments = assign_risk_segment(
            proba, self.bundle.segment_settings, self.bundle.decision_threshold
        )
        out["risk_segment"] = segments["risk_segment"].to_numpy()
        out["risk_label"] = segments["risk_label"].to_numpy()
        return out

    def predict_one(self, record: Mapping) -> dict:
        """Score a single customer and return a plain ``dict`` (API-friendly)."""
        result = self.predict(record).iloc[0].to_dict()
        result["default_probability"] = float(result["default_probability"])
        result["predicted_label"] = int(result["predicted_label"])
        result["model_version"] = self.model_version
        return result
