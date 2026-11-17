"""Persisting and loading trained-model artifacts.

``model_bundle.joblib`` is the single source of truth for inference: it holds
the fitted pipeline together with everything needed to turn a raw score into a
decision (prior correction, threshold, segment settings, feature lists).
``preprocessor.joblib`` and ``classifier.joblib`` are convenience extracts, and
``model_metadata.json`` is human/DB readable (feeds the SQL ``model_registry``).
"""

from __future__ import annotations

import logging
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from imblearn.pipeline import Pipeline as ImbPipeline

from ml.config import DATASET_VERSION, PATHS, Paths, SegmentSettings
from ml.imbalance import PriorCorrection
from utils.io_utils import save_json

logger = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    """Everything required to score a customer, serialised as one object."""

    model_name: str
    model_version: str
    pipeline: ImbPipeline
    prior_correction: PriorCorrection
    decision_threshold: float
    segment_settings: SegmentSettings
    numeric_features: list[str]
    categorical_features: list[str]
    metadata: dict = field(default_factory=dict)


def make_model_version(model_name: str, when: datetime | None = None) -> str:
    """Version string like ``random_forest-20261115T101500Z``."""
    when = when or datetime.now(timezone.utc)
    return f"{model_name}-{when.strftime('%Y%m%dT%H%M%SZ')}"


def build_metadata(
    bundle_inputs: dict,
    *,
    seed: int,
    train_rows: int,
    metrics: dict,
    dataset_version: str = DATASET_VERSION,
) -> dict:
    """Assemble the JSON metadata stored beside the model."""
    return {
        **bundle_inputs,
        "dataset_version": dataset_version,
        "random_state": seed,
        "train_rows": train_rows,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": metrics,
        "library_versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }


def save_bundle(bundle: ModelBundle, paths: Paths = PATHS) -> dict[str, Path]:
    """Write the bundle, its extracts and metadata into ``saved_models/``.

    Returns:
        ``{"bundle", "preprocessor", "classifier", "metadata"}`` -> file path.
    """
    paths.saved_models.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, paths.model_bundle)
    joblib.dump(bundle.pipeline.named_steps["preprocessor"], paths.preprocessor_artifact)
    joblib.dump(bundle.pipeline.named_steps["model"], paths.classifier_artifact)
    save_json(
        {
            "model_name": bundle.model_name,
            "model_version": bundle.model_version,
            "decision_threshold": bundle.decision_threshold,
            "prior_correction": asdict(bundle.prior_correction),
            "segment_settings": asdict(bundle.segment_settings),
            "numeric_features": bundle.numeric_features,
            "categorical_features": bundle.categorical_features,
            **bundle.metadata,
        },
        paths.model_metadata,
    )
    logger.info("Saved model artifacts to %s", paths.saved_models)
    return {
        "bundle": paths.model_bundle,
        "preprocessor": paths.preprocessor_artifact,
        "classifier": paths.classifier_artifact,
        "metadata": paths.model_metadata,
    }


def load_bundle(path: Path | str | None = None) -> ModelBundle:
    """Load a :class:`ModelBundle` saved by :func:`save_bundle`.

    Raises:
        FileNotFoundError: If the bundle file does not exist.
    """
    path = Path(path) if path else PATHS.model_bundle
    if not path.is_file():
        raise FileNotFoundError(f"Model bundle not found: {path}. Run `python -m ml.train` first.")
    bundle = joblib.load(path)
    logger.info("Loaded model bundle %s from %s", bundle.model_version, path)
    return bundle
