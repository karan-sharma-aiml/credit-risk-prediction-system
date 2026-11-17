"""Central configuration: paths, schema definition and domain constants.

Every tunable value used by the data pipeline lives here so that nothing is
hardcoded inside functions or notebooks. Later parts (training, API) import the
same schema constants, which keeps training and serving consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
RANDOM_STATE: Final[int] = 42


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Paths:
    """Resolved project directories, derived from a single root."""

    root: Path = PROJECT_ROOT

    @classmethod
    def from_root(cls, root: Path | str) -> "Paths":
        """Build a ``Paths`` rooted somewhere else (used heavily in tests)."""
        return cls(root=Path(root))

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_interim(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    @property
    def summaries(self) -> Path:
        return self.reports / "summaries"

    @property
    def saved_models(self) -> Path:
        return self.root / "saved_models"

    @property
    def raw_dataset(self) -> Path:
        return self.data_raw / RAW_DATASET_NAME

    @property
    def clean_dataset(self) -> Path:
        return self.data_interim / CLEAN_DATASET_NAME

    @property
    def feature_dataset(self) -> Path:
        return self.data_processed / FEATURE_DATASET_NAME

    @property
    def model_bundle(self) -> Path:
        return self.saved_models / MODEL_BUNDLE_NAME

    @property
    def preprocessor_artifact(self) -> Path:
        return self.saved_models / PREPROCESSOR_NAME

    @property
    def classifier_artifact(self) -> Path:
        return self.saved_models / CLASSIFIER_NAME

    @property
    def model_metadata(self) -> Path:
        return self.saved_models / MODEL_METADATA_NAME

    def ensure(self) -> None:
        """Create every output directory that the pipeline writes to."""
        for path in (
            self.data_raw,
            self.data_interim,
            self.data_processed,
            self.figures,
            self.summaries,
            self.saved_models,
        ):
            path.mkdir(parents=True, exist_ok=True)


RAW_DATASET_NAME: Final[str] = "credit_risk_raw.csv"
CLEAN_DATASET_NAME: Final[str] = "credit_risk_clean.csv"
FEATURE_DATASET_NAME: Final[str] = "credit_risk_features.csv"
DATASET_VERSION: Final[str] = "synthetic-v1"

MODEL_BUNDLE_NAME: Final[str] = "model_bundle.joblib"
PREPROCESSOR_NAME: Final[str] = "preprocessor.joblib"
CLASSIFIER_NAME: Final[str] = "classifier.joblib"
MODEL_METADATA_NAME: Final[str] = "model_metadata.json"

PATHS: Final[Paths] = Paths()


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
ID_COL: Final[str] = "customer_id"
TARGET_COL: Final[str] = "default"

NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "age",
    "annual_income",
    "employment_years",
    "loan_amount",
    "loan_term_months",
    "credit_score",
    "monthly_debt_payment",
    "credit_limit",
    "revolving_balance",
    "savings_balance",
    "monthly_expenses",
    "num_existing_loans",
    "num_late_payments",
)

CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = (
    "employment_type",
    "repayment_history",
    "home_ownership",
)

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    ID_COL,
    *NUMERIC_COLUMNS,
    *CATEGORICAL_COLUMNS,
    TARGET_COL,
)

# Allowed values for categorical columns (after lower-casing / snake_casing).
CATEGORY_LEVELS: Final[dict[str, tuple[str, ...]]] = {
    "employment_type": ("salaried", "self_employed", "retired", "unemployed"),
    "repayment_history": ("excellent", "good", "fair", "poor"),
    "home_ownership": ("rent", "mortgage", "own", "other"),
}

# Hard business-validity bounds (min, max); ``None`` means unbounded on that
# side. Values outside the range are treated as data-entry errors and set to
# NaN by the cleaning step (they are then imputed inside the sklearn pipeline).
VALID_RANGES: Final[dict[str, tuple[float | None, float | None]]] = {
    "age": (18, 100),
    "annual_income": (1, None),
    "employment_years": (0, 60),
    "loan_amount": (1, None),
    "loan_term_months": (1, 480),
    "credit_score": (300, 850),
    "monthly_debt_payment": (0, None),
    "credit_limit": (0, None),
    "revolving_balance": (0, None),
    "savings_balance": (0, None),
    "monthly_expenses": (0, None),
    "num_existing_loans": (0, 50),
    "num_late_payments": (0, 100),
}

# Accepted spellings of the target, mapped to {1: default, 0: repaid}.
TARGET_LABEL_MAP: Final[dict[str, int]] = {
    "1": 1, "1.0": 1, "yes": 1, "y": 1, "true": 1, "t": 1,
    "default": 1, "defaulted": 1, "bad": 1,
    "0": 0, "0.0": 0, "no": 0, "n": 0, "false": 0, "f": 0,
    "non-default": 0, "non_default": 0, "good": 0, "paid": 0, "repaid": 0,
}


# --------------------------------------------------------------------------- #
# Feature-engineering settings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureSettings:
    """Tunable constants for engineered features.

    Buckets use *fixed, business-defined* edges (not data quantiles) so that
    the transformation is stateless: a customer receives the same bucket in
    training, batch scoring and the future API, and no information leaks from
    the evaluation set.
    """

    age_bins: tuple[float, ...] = (18, 25, 35, 45, 55, 65, np.inf)
    age_labels: tuple[str, ...] = ("18-24", "25-34", "35-44", "45-54", "55-64", "65+")

    income_bins: tuple[float, ...] = (0, 25_000, 50_000, 80_000, 120_000, np.inf)
    income_labels: tuple[str, ...] = ("low", "lower_mid", "mid", "upper_mid", "high")

    # Caps stop a single tiny denominator from producing absurd ratios.
    dti_cap: float = 5.0
    loan_to_income_cap: float = 10.0
    expense_ratio_cap: float = 5.0
    savings_ratio_cap: float = 20.0
    savings_runway_cap_months: float = 60.0
    utilization_cap: float = 1.5

    high_utilization_threshold: float = 0.70
    stability_tenure_cap_years: float = 10.0
    # Heuristic weight for how "stable" each employment type is (0-1).
    employment_type_weights: tuple[tuple[str, float], ...] = (
        ("salaried", 1.0),
        ("retired", 0.9),
        ("self_employed", 0.7),
        ("unemployed", 0.0),
    )


FEATURE_SETTINGS: Final[FeatureSettings] = FeatureSettings()

# Names of engineered columns (kept here so Part 2 can build column lists
# without importing feature logic).
ENGINEERED_NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "debt_to_income",
    "loan_to_income",
    "loan_payment_to_income",
    "expense_ratio",
    "savings_ratio",
    "savings_runway_months",
    "credit_utilization",
    "high_utilization_flag",
    "employment_stability_score",
    "has_late_payments",
)
ENGINEERED_CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = ("age_group", "income_bucket")


# --------------------------------------------------------------------------- #
# Part 2: training, selection and segmentation settings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SplitSettings:
    """Stratified train / validation / test split (fractions of the full data)."""

    test_size: float = 0.20
    validation_size: float = 0.20


@dataclass(frozen=True)
class BalancingSettings:
    """SMOTE configuration (applied to the training split only).

    ``sampling_strategy`` is the desired minority/majority ratio after
    resampling (0.5 -> one default for every two repaid loans). A partial
    rebalance keeps synthetic data to a minimum while still removing the
    model's incentive to ignore defaulters.
    """

    enabled: bool = True
    sampling_strategy: float = 0.5
    k_neighbors: int = 5


@dataclass(frozen=True)
class TuningSettings:
    """Lightweight randomized search (cross-validation on training data only)."""

    n_iter: int = 8
    cv_folds: int = 3
    scoring: str = "average_precision"


@dataclass(frozen=True)
class SelectionSettings:
    """Business rule used to pick the champion model (on validation data).

    * Decision threshold per model: maximise F-beta (beta > 1 favours recall,
      because missing a defaulter costs more than a false alarm) subject to a
      minimum precision, so "flag everyone" is never optimal.
    * Model score = weighted sum of validation recall, F1 and ROC-AUC.
    """

    threshold_beta: float = 2.0
    min_precision: float = 0.25
    weights: tuple[tuple[str, float], ...] = (
        ("recall", 0.40),
        ("f1", 0.30),
        ("roc_auc", 0.30),
    )


@dataclass(frozen=True)
class SegmentSettings:
    """Probability cut-offs for business risk segments.

    ``p < low_max`` -> low; ``low_max <= p < high_min`` -> medium;
    ``p >= high_min`` -> high. Probabilities are corrected for the resampling
    prior shift, so they approximate real-world default rates (portfolio
    average is roughly 14%).
    """

    low_max: float = 0.10
    high_min: float = 0.30
    codes: tuple[str, ...] = ("low", "medium", "high")
    labels: tuple[str, ...] = ("Low Risk", "Medium Risk", "High Risk")


SPLIT_SETTINGS: Final[SplitSettings] = SplitSettings()
BALANCING_SETTINGS: Final[BalancingSettings] = BalancingSettings()
TUNING_SETTINGS: Final[TuningSettings] = TuningSettings()
SELECTION_SETTINGS: Final[SelectionSettings] = SelectionSettings()
SEGMENT_SETTINGS: Final[SegmentSettings] = SegmentSettings()
