"""End-to-end Part 1 data pipeline: load -> validate -> clean -> engineer -> EDA.

Run from the project root::

    python -m ml.pipeline                 # generate synthetic data if needed
    python -m ml.pipeline --n-rows 20000  # bigger synthetic portfolio
    python -m ml.pipeline --train         # data pipeline + model training (or `python -m ml.train`)
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ml.config import (
    CATEGORICAL_COLUMNS,
    ENGINEERED_CATEGORICAL_COLUMNS,
    ENGINEERED_NUMERIC_COLUMNS,
    NUMERIC_COLUMNS,
    PATHS,
    RANDOM_STATE,
    Paths,
)
from ml.data import DataQualityReport, get_or_create_raw_dataset, summarize_dataset, validate_columns
from ml.eda import run_eda
from ml.features import engineer_features
from ml.preprocessing import CleaningReport, clean_dataset
from utils.io_utils import save_json
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Artefacts produced by :func:`run_data_pipeline`."""

    clean_df: pd.DataFrame
    features_df: pd.DataFrame
    quality_report: DataQualityReport
    cleaning_report: CleaningReport
    clean_path: Path
    features_path: Path


def get_feature_lists() -> tuple[list[str], list[str]]:
    """Return ``(numeric, categorical)`` model-input columns (base + engineered).

    Part 2 uses this to build the sklearn preprocessor without hardcoding names.
    """
    numeric = [*NUMERIC_COLUMNS, *ENGINEERED_NUMERIC_COLUMNS]
    categorical = [*CATEGORICAL_COLUMNS, *ENGINEERED_CATEGORICAL_COLUMNS]
    return numeric, categorical


def run_data_pipeline(
    *,
    paths: Paths = PATHS,
    n_rows: int = 10_000,
    seed: int = RANDOM_STATE,
    run_eda_step: bool = True,
) -> PipelineResult:
    """Execute the full data pipeline and persist its outputs.

    Args:
        paths: Project directories (override in tests).
        n_rows: Rows to generate if the raw file does not exist yet.
        seed: Seed for synthetic generation.
        run_eda_step: Skip EDA plots/tables when ``False`` (faster tests).

    Returns:
        A :class:`PipelineResult` with data frames, reports and file paths.
    """
    paths.ensure()

    raw = get_or_create_raw_dataset(paths.raw_dataset, n_rows=n_rows, seed=seed)
    validate_columns(raw)
    quality = summarize_dataset(raw)
    save_json(quality.to_dict(), paths.summaries / "data_quality_raw.json")

    clean, cleaning = clean_dataset(raw)
    save_json(cleaning.to_dict(), paths.summaries / "cleaning_report.json")
    clean.to_csv(paths.clean_dataset, index=False)

    features = engineer_features(clean)
    features.to_csv(paths.feature_dataset, index=False)
    logger.info("Saved datasets -> %s, %s", paths.clean_dataset, paths.feature_dataset)

    if run_eda_step:
        run_eda(features, figures_dir=paths.figures, summaries_dir=paths.summaries)

    return PipelineResult(clean, features, quality, cleaning, paths.clean_dataset, paths.feature_dataset)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run the credit-risk data pipeline.")
    parser.add_argument("--n-rows", type=int, default=10_000, help="Synthetic rows if raw file is absent")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--skip-eda", action="store_true")
    parser.add_argument("--train", action="store_true", help="Also run model training (ml.train)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    run_data_pipeline(n_rows=args.n_rows, seed=args.seed, run_eda_step=not args.skip_eda)
    if args.train:
        from ml.train import run_training_pipeline  # lazy: train imports this module

        run_training_pipeline(n_rows=args.n_rows, seed=args.seed)


if __name__ == "__main__":
    main()
