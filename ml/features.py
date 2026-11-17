"""Financial feature engineering for credit-risk modelling.

All features are *stateless* functions of a single row, so they can be applied
identically in training, batch scoring and a future API without leakage.
Missing inputs propagate to NaN (never to 0), and the downstream imputer
decides how to fill them.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ml.config import FEATURE_SETTINGS, FeatureSettings

logger = logging.getLogger(__name__)

#: Why each feature matters for credit risk (also rendered in docs/).
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "debt_to_income": (
        "Monthly debt payments / monthly income. The core affordability metric; "
        "lenders typically get cautious above ~36-43%."
    ),
    "loan_to_income": (
        "Requested loan / annual income. Measures how large the new obligation is "
        "relative to earning power, independent of existing debt."
    ),
    "loan_payment_to_income": (
        "(Loan amount / term in months) / monthly income. Principal-only proxy "
        "for the new monthly burden (interest ignored); uses the loan term."
    ),
    "expense_ratio": (
        "Monthly living expenses / monthly income. High values mean little "
        "disposable income to absorb a new repayment."
    ),
    "savings_ratio": (
        "Savings / annual income. A buffer indicator: savings reduce default "
        "probability when income is disrupted."
    ),
    "savings_runway_months": (
        "Savings / monthly expenses. How many months the customer can cover "
        "expenses without income."
    ),
    "credit_utilization": (
        "Revolving balance / credit limit. A bureau-style signal of credit "
        "stress; high utilisation is a strong default predictor."
    ),
    "high_utilization_flag": (
        "1 if utilisation >= threshold (default 70%). A non-linear risk tier "
        "that simple models can use directly."
    ),
    "employment_stability_score": (
        "Employment-type weight x tenure (capped at 10 years), range 0-1. "
        "Proxy for income continuity; weights are heuristic assumptions."
    ),
    "has_late_payments": (
        "1 if any late payment in the observation window. Past delinquency is "
        "one of the strongest predictors of future default."
    ),
    "age_group": (
        "Fixed age bands. Captures non-linear life-stage effects "
        "(e.g. thin-file young borrowers). Use with care for fairness reviews."
    ),
    "income_bucket": (
        "Fixed income bands (low ... high). Supports segmentation and "
        "non-linear income effects."
    ),
}


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, returning NaN when the denominator is zero, negative or missing."""
    return numerator / denominator.where(denominator > 0)


def _flag(condition: pd.Series, source: pd.Series) -> pd.Series:
    """0/1 float flag that stays NaN where ``source`` is missing."""
    return condition.astype(float).where(source.notna())


def engineer_features(
    df: pd.DataFrame,
    settings: FeatureSettings = FEATURE_SETTINGS,
) -> pd.DataFrame:
    """Add engineered financial features to a cleaned DataFrame.

    Args:
        df: Cleaned data containing the base schema columns.
        settings: Caps, thresholds and bucket edges.

    Returns:
        A copy of ``df`` with the engineered columns appended.
    """
    out = df.copy()
    monthly_income = out["annual_income"] / 12.0

    out["debt_to_income"] = _safe_divide(out["monthly_debt_payment"], monthly_income).clip(
        0, settings.dti_cap
    )
    out["loan_to_income"] = _safe_divide(out["loan_amount"], out["annual_income"]).clip(
        0, settings.loan_to_income_cap
    )
    monthly_loan_principal = _safe_divide(out["loan_amount"], out["loan_term_months"])
    out["loan_payment_to_income"] = _safe_divide(monthly_loan_principal, monthly_income).clip(
        0, settings.dti_cap
    )
    out["expense_ratio"] = _safe_divide(out["monthly_expenses"], monthly_income).clip(
        0, settings.expense_ratio_cap
    )
    out["savings_ratio"] = _safe_divide(out["savings_balance"], out["annual_income"]).clip(
        0, settings.savings_ratio_cap
    )
    out["savings_runway_months"] = _safe_divide(
        out["savings_balance"], out["monthly_expenses"]
    ).clip(0, settings.savings_runway_cap_months)

    utilization = _safe_divide(out["revolving_balance"], out["credit_limit"]).clip(
        0, settings.utilization_cap
    )
    out["credit_utilization"] = utilization
    out["high_utilization_flag"] = _flag(
        utilization >= settings.high_utilization_threshold, utilization
    )

    weights = pd.Series(dict(settings.employment_type_weights))
    tenure = (out["employment_years"] / settings.stability_tenure_cap_years).clip(0, 1)
    out["employment_stability_score"] = out["employment_type"].map(weights) * tenure

    out["has_late_payments"] = _flag(out["num_late_payments"] > 0, out["num_late_payments"])

    out["age_group"] = pd.cut(
        out["age"], bins=list(settings.age_bins), labels=list(settings.age_labels), right=False
    ).astype(object)
    out["income_bucket"] = pd.cut(
        out["annual_income"],
        bins=list(settings.income_bins),
        labels=list(settings.income_labels),
        right=False,
    ).astype(object)

    out = out.replace([np.inf, -np.inf], np.nan)
    logger.info("Engineered %d new features", out.shape[1] - df.shape[1])
    return out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """sklearn wrapper around :func:`engineer_features` for use in ``Pipeline``."""

    def __init__(self, settings: FeatureSettings = FEATURE_SETTINGS) -> None:
        self.settings = settings

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return engineer_features(X, self.settings)
