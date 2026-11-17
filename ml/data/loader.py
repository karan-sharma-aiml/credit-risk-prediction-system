"""Dataset loading with schema validation and optional column renaming."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ml.config import PATHS, RANDOM_STATE
from ml.data.generator import generate_synthetic_credit_data
from ml.data.validation import validate_columns

logger = logging.getLogger(__name__)


def load_dataset(
    path: Path | str,
    *,
    column_map: dict[str, str] | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Load a credit-risk CSV into a DataFrame.

    Args:
        path: Location of the CSV file.
        column_map: Optional ``{source_name: schema_name}`` renames, so a real
            dataset with different headers can be plugged in without code
            changes elsewhere.
        validate: If ``True``, raise when required columns are missing.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        SchemaValidationError: If validation is on and columns are missing.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if column_map:
        df = df.rename(columns=column_map)
    logger.info("Loaded %s -> %d rows x %d columns", path.name, *df.shape)

    if validate:
        validate_columns(df)
    return df


def get_or_create_raw_dataset(
    path: Path | str | None = None,
    *,
    n_rows: int = 10_000,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Load the raw dataset, generating and saving the synthetic one if absent.

    Args:
        path: CSV location; defaults to ``PATHS.raw_dataset``.
        n_rows: Rows to generate when the file does not exist.
        seed: Seed for generation.
    """
    path = Path(path) if path else PATHS.raw_dataset
    if path.is_file():
        return load_dataset(path)

    logger.info("Raw dataset missing; generating synthetic data at %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_credit_data(n_rows=n_rows, seed=seed)
    df.to_csv(path, index=False)
    return df
