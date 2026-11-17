"""Shared fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.config import Paths
from ml.data import generate_synthetic_credit_data
from ml.preprocessing import clean_dataset
from ml.train import TrainingResult, run_training_pipeline


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """Small messy synthetic dataset (deterministic)."""
    return generate_synthetic_credit_data(n_rows=600, seed=7)


@pytest.fixture(scope="session")
def clean_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Cleaned version of ``raw_df``."""
    cleaned, _ = clean_dataset(raw_df)
    return cleaned


@pytest.fixture(scope="session")
def trained_run(tmp_path_factory: pytest.TempPathFactory) -> TrainingResult:
    """One full training run (all available models) on a small dataset."""
    paths = Paths.from_root(tmp_path_factory.mktemp("project"))
    return run_training_pipeline(paths=paths, n_rows=2500, seed=11, run_plots=True)
