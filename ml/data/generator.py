"""Reproducible synthetic credit-risk dataset generator.

No single public dataset contains every field required by the business brief
(income, debt, savings, expenses, credit score, late payments, ...). This
module generates a realistic *synthetic* portfolio so the whole pipeline can
be built and tested end-to-end. The loader accepts any real dataset with the
same schema (see ``ml.data.loader.load_dataset(column_map=...)``).

IMPORTANT: the default label is produced from a hand-written risk formula, so
model scores on this data show that the pipeline works, not how a model would
perform on real borrowers.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ml.config import CATEGORY_LEVELS, ID_COL, RANDOM_STATE, TARGET_COL

logger = logging.getLogger(__name__)

_TARGET_DEFAULT_RATE = 0.14


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _calibrate_intercept(logits: np.ndarray, target_rate: float) -> float:
    """Find the intercept shift so that the mean probability hits ``target_rate``."""
    low, high = -15.0, 15.0
    for _ in range(60):
        mid = (low + high) / 2.0
        if _sigmoid(logits + mid).mean() > target_rate:
            high = mid
        else:
            low = mid
    return (low + high) / 2.0


def _draw_home_ownership(age: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    p_rent = np.clip(0.75 - 0.012 * (age - 18), 0.15, 0.75)
    p_other = 0.04
    remaining = 1.0 - p_rent - p_other
    p_mortgage = remaining * 0.6
    u = rng.random(age.size)
    return np.select(
        [u < p_rent, u < p_rent + p_mortgage, u < p_rent + p_mortgage + remaining * 0.4],
        ["rent", "mortgage", "own"],
        default="other",
    )


def _build_clean_portfolio(n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    """Create an internally consistent, issue-free portfolio."""
    age = np.clip(rng.normal(41, 12, n_rows), 18, 80).round().astype(int)

    employment = rng.choice(
        ["salaried", "self_employed", "unemployed", "retired"],
        size=n_rows,
        p=[0.62, 0.22, 0.06, 0.10],
    )
    employment = np.where((employment == "retired") & (age < 58), "salaried", employment)
    employment = np.where((age >= 65) & (rng.random(n_rows) < 0.6), "retired", employment)

    max_years = np.clip(age - 20, 0, None)
    employment_years = rng.random(n_rows) ** 0.8 * max_years * 0.7
    employment_years = np.where(
        employment == "unemployed", rng.uniform(0, 0.5, n_rows), employment_years
    ).round(1)

    type_multiplier = pd.Series(employment).map(
        {"salaried": 1.0, "self_employed": 1.1, "unemployed": 0.25, "retired": 0.6}
    ).to_numpy()
    age_effect = 1.0 + 0.012 * (np.minimum(age, 55) - 30)
    annual_income = np.clip(
        rng.lognormal(mean=10.9, sigma=0.55, size=n_rows) * type_multiplier * age_effect,
        8_000,
        400_000,
    ).round(2)
    monthly_income = annual_income / 12.0

    credit_score = np.clip(
        650 + 0.9 * (age - 40) + 20 * np.log(annual_income / 50_000) + rng.normal(0, 55, n_rows),
        300,
        850,
    ).round().astype(int)

    home = _draw_home_ownership(age, rng)

    loan_amount = np.clip(
        annual_income * rng.beta(2, 4, n_rows) * 1.2, 1_000, 150_000
    ).round(-2)
    loan_term = rng.choice([12, 24, 36, 48, 60, 72, 84], size=n_rows,
                           p=[0.05, 0.10, 0.25, 0.20, 0.25, 0.10, 0.05])

    num_loans = np.clip(rng.poisson(1.2, n_rows), 0, 8)
    mortgage_component = (home == "mortgage") * monthly_income * rng.uniform(0.12, 0.25, n_rows)
    monthly_debt = (
        num_loans * monthly_income * rng.uniform(0.03, 0.08, n_rows) + mortgage_component
    ).round(2)

    credit_limit = np.maximum(monthly_income * rng.uniform(1.5, 6.0, n_rows), 500).round(-1)
    utilization = np.clip(rng.beta(2, 3, n_rows) + (650 - credit_score) / 1000.0, 0.0, 1.2)
    revolving_balance = (credit_limit * utilization).round(2)

    monthly_expenses = np.maximum(monthly_income * rng.uniform(0.35, 0.85, n_rows), 300).round(2)

    savings = annual_income * rng.exponential(0.25, n_rows) * (1 + (age - 18) / 60.0)
    savings = np.where(rng.random(n_rows) < 0.15, 0.0, savings).round(2)

    lam = np.clip(0.3 + (700 - credit_score) / 150.0, 0.05, None) + 0.5 * (utilization > 0.8)
    late_payments = np.clip(rng.poisson(lam), 0, 12)

    level = np.select([late_payments == 0, late_payments <= 1, late_payments <= 3], [0, 1, 2], 3)
    level = np.clip(level + rng.choice([-1, 0, 0, 0, 0, 0, 0, 0, 0, 1], size=n_rows), 0, 3)
    repayment = np.array(CATEGORY_LEVELS["repayment_history"])[level]

    # --- Latent default risk ------------------------------------------------
    dti = monthly_debt / monthly_income
    lti = loan_amount / annual_income
    exp_ratio = monthly_expenses / monthly_income
    sav_ratio = savings / annual_income
    logits = (
        3.0 * dti
        + 1.2 * lti
        + 1.0 * np.clip(utilization, 0, 1.2)
        + 0.35 * late_payments
        - 0.012 * (credit_score - 650)
        - 1.5 * np.clip(sav_ratio, 0, 1)
        + 1.0 * (employment == "unemployed")
        + 0.3 * (employment == "self_employed")
        - 0.04 * np.minimum(employment_years, 15)
        + 0.5 * (home == "rent")
        + 0.8 * (exp_ratio > 0.8)
        + rng.normal(0, 0.5, n_rows)
    )
    logits = logits + _calibrate_intercept(logits, _TARGET_DEFAULT_RATE)
    default = (rng.random(n_rows) < _sigmoid(logits)).astype(int)

    return pd.DataFrame(
        {
            ID_COL: [f"CUST{i:07d}" for i in range(1, n_rows + 1)],
            "age": age,
            "annual_income": annual_income,
            "employment_type": employment,
            "employment_years": employment_years,
            "loan_amount": loan_amount,
            "loan_term_months": loan_term,
            "credit_score": credit_score,
            "monthly_debt_payment": monthly_debt,
            "credit_limit": credit_limit,
            "revolving_balance": revolving_balance,
            "savings_balance": savings,
            "monthly_expenses": monthly_expenses,
            "num_existing_loans": num_loans,
            "num_late_payments": late_payments,
            "repayment_history": repayment,
            "home_ownership": home,
            TARGET_COL: default,
        }
    )


def _inject_data_issues(
    df: pd.DataFrame,
    rng: np.random.Generator,
    missing_rate: float,
    duplicate_rate: float,
    outlier_rate: float,
) -> pd.DataFrame:
    """Add the messiness a real extract would have, so cleaning has work to do."""
    df = df.copy()
    n = len(df)

    for col in (
        "employment_years", "credit_score", "savings_balance", "monthly_expenses",
        "credit_limit", "home_ownership", "repayment_history",
    ):
        df.loc[rng.random(n) < missing_rate, col] = np.nan

    # Extreme but plausible-looking values (handled by quantile clipping).
    outlier_idx = rng.random(n) < outlier_rate
    df.loc[outlier_idx, "annual_income"] = df.loc[outlier_idx, "annual_income"] * 20
    outlier_idx = rng.random(n) < outlier_rate
    df.loc[outlier_idx, "loan_amount"] = df.loc[outlier_idx, "loan_amount"] * 10

    # Impossible values (handled by domain-rule validation).
    df.loc[rng.random(n) < outlier_rate / 2, "age"] = 150
    df.loc[rng.random(n) < outlier_rate / 2, "savings_balance"] = -500.0

    # A handful of unlabeled rows.
    df[TARGET_COL] = df[TARGET_COL].astype(float)
    df.loc[rng.random(n) < 0.001, TARGET_COL] = np.nan

    n_dupes = int(n * duplicate_rate)
    if n_dupes:
        df = pd.concat([df, df.sample(n_dupes, random_state=int(rng.integers(1e6)))])

    return df.sample(frac=1.0, random_state=int(rng.integers(1e6))).reset_index(drop=True)


def generate_synthetic_credit_data(
    n_rows: int = 10_000,
    seed: int = RANDOM_STATE,
    missing_rate: float = 0.03,
    duplicate_rate: float = 0.005,
    outlier_rate: float = 0.004,
) -> pd.DataFrame:
    """Generate a messy synthetic credit-risk dataset.

    Args:
        n_rows: Number of unique customers before duplicates are added.
        seed: Seed for full reproducibility.
        missing_rate: Per-column probability of a missing value (selected columns).
        duplicate_rate: Fraction of rows duplicated verbatim.
        outlier_rate: Probability of extreme / invalid values per affected column.

    Returns:
        A raw-looking DataFrame that follows ``ml.config.REQUIRED_COLUMNS``.
    """
    if n_rows < 50:
        raise ValueError("n_rows must be at least 50 for a meaningful dataset.")
    rng = np.random.default_rng(seed)
    clean = _build_clean_portfolio(n_rows, rng)
    raw = _inject_data_issues(clean, rng, missing_rate, duplicate_rate, outlier_rate)
    logger.info(
        "Generated synthetic dataset: %d rows (default rate before noise: %.1f%%)",
        len(raw), clean[TARGET_COL].mean() * 100,
    )
    return raw
