"""Class-imbalance handling (training data only).

Why balancing matters in credit risk
------------------------------------
Defaults are rare (~14% here, often 1-5% in production). A classifier that
minimises overall error can reach high accuracy by approving everyone, which
is exactly the costly behaviour a lender must avoid: a missed defaulter loses
principal, a wrongly declined good customer only loses margin. Oversampling the
minority class during training gives the model enough signal to learn the
defaulter profile.

Two rules keep this honest:

1. SMOTE runs *inside* the sklearn/imblearn ``Pipeline`` after the
   preprocessor, so it only ever sees training rows (also within each
   cross-validation fold) and is skipped automatically at predict time.
2. Resampling distorts the class prior the model sees, inflating predicted
   probabilities. :class:`PriorCorrection` undoes that shift so outputs can be
   read as real-world default probabilities (needed for risk segments).

Limitation: SMOTE interpolates numeric columns, including one-hot columns, so
synthetic rows may have fractional category indicators. This is a common,
accepted approximation; SMOTENC is a possible later refinement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer

from ml.config import BALANCING_SETTINGS, RANDOM_STATE, BalancingSettings


def build_smote(settings: BalancingSettings = BALANCING_SETTINGS, seed: int = RANDOM_STATE) -> SMOTE:
    """Create the configured SMOTE sampler."""
    return SMOTE(
        sampling_strategy=settings.sampling_strategy,
        k_neighbors=settings.k_neighbors,
        random_state=seed,
    )


def build_training_pipeline(
    preprocessor: ColumnTransformer,
    model: BaseEstimator,
    *,
    balance: bool = True,
    settings: BalancingSettings = BALANCING_SETTINGS,
    seed: int = RANDOM_STATE,
) -> ImbPipeline:
    """Assemble ``preprocessor -> [SMOTE] -> model`` as one leakage-safe pipeline.

    Step names are fixed (``preprocessor``, ``smote``, ``model``) so tuning
    grids, artifact extraction and tests can address them reliably.
    """
    steps: list[tuple[str, object]] = [("preprocessor", preprocessor)]
    if balance and settings.enabled:
        steps.append(("smote", build_smote(settings, seed)))
    steps.append(("model", model))
    return ImbPipeline(steps)


def expected_resampled_counts(y_train: pd.Series, sampling_strategy: float) -> dict[str, int]:
    """Class counts SMOTE will produce for ``y_train`` (without running it)."""
    n_pos, n_neg = int(y_train.sum()), int((y_train == 0).sum())
    n_pos_after = max(n_pos, int(sampling_strategy * n_neg))  # imblearn truncates
    return {"n_repaid": n_neg, "n_default": n_pos_after}


def balancing_report(
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    *,
    balance: bool = True,
    settings: BalancingSettings = BALANCING_SETTINGS,
) -> pd.DataFrame:
    """Class distribution before/after balancing for every split.

    Validation and test rows always show ``after == before`` because they are
    never resampled.
    """
    after_train = (
        expected_resampled_counts(y_train, settings.sampling_strategy)
        if balance and settings.enabled
        else {"n_repaid": int((y_train == 0).sum()), "n_default": int(y_train.sum())}
    )
    rows = []
    for name, y in (("train", y_train), ("validation", y_val), ("test", y_test)):
        before = {"n_repaid": int((y == 0).sum()), "n_default": int(y.sum())}
        after = after_train if name == "train" else before
        rows.append({
            "split": name,
            "repaid_before": before["n_repaid"],
            "default_before": before["n_default"],
            "default_rate_before": round(before["n_default"] / sum(before.values()), 4),
            "repaid_after": after["n_repaid"],
            "default_after": after["n_default"],
            "default_rate_after": round(after["n_default"] / sum(after.values()), 4),
        })
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class PriorCorrection:
    """Re-weights probabilities from the resampled prior back to the true prior.

    For a model trained at positive-class share ``train_prior`` but deployed
    where the share is ``true_prior``, the corrected odds are
    ``odds_model * [true_prior / (1 - true_prior)] / [train_prior / (1 - train_prior)]``.
    The map is monotonic, so ROC-AUC and PR-AUC are unaffected.
    """

    train_prior: float
    true_prior: float

    @classmethod
    def identity(cls, prior: float = 0.5) -> "PriorCorrection":
        """No-op correction (used when no resampling was applied)."""
        return cls(prior, prior)

    @classmethod
    def from_counts(
        cls, y_train: pd.Series, *, balance: bool, settings: BalancingSettings = BALANCING_SETTINGS
    ) -> "PriorCorrection":
        """Derive the correction from the training labels and balancing config."""
        true_prior = float(y_train.mean())
        if not (balance and settings.enabled):
            return cls.identity(true_prior)
        counts = expected_resampled_counts(y_train, settings.sampling_strategy)
        return cls(counts["n_default"] / (counts["n_default"] + counts["n_repaid"]), true_prior)

    def apply(self, probabilities: np.ndarray) -> np.ndarray:
        """Return prior-corrected probabilities."""
        p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        factor = (self.true_prior / (1 - self.true_prior)) / (self.train_prior / (1 - self.train_prior))
        odds = p / (1 - p) * factor
        return odds / (1 + odds)
