"""Recommendation engine: risk segment -> business action (+ rule-based considerations).

Two independent, replaceable pieces:

* **Policy** - maps a risk segment to an action (Approve / Manual Review / Reject).
* **Rules** - small checks on derived financial ratios that add human-readable
  considerations for the underwriter. They never change the action.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ml.config import FEATURE_SETTINGS
from ml.features import engineer_features
from ml.preprocessing import clean_for_inference

Features = Mapping[str, float | None]


@dataclass(frozen=True)
class Recommendation:
    """Recommendation returned with every prediction."""

    action: str
    title: str
    message: str
    considerations: list[str] = field(default_factory=list)


#: segment code -> (action, title, message)
DEFAULT_POLICY: dict[str, tuple[str, str, str]] = {
    "low": ("approve", "Approve Loan", "Low default risk. Proceed with standard approval and terms."),
    "medium": (
        "manual_review",
        "Manual Review",
        "Moderate default risk. Route to an underwriter; consider adjusted terms or additional documents.",
    ),
    "high": ("reject", "Reject Loan", "High default risk. Decline, or offer a reduced amount with collateral."),
}


@dataclass(frozen=True)
class RiskRule:
    """A named predicate over derived features plus the message shown when it fires."""

    name: str
    applies: Callable[[Features], bool]
    message: str


def _gt(key: str, limit: float) -> Callable[[Features], bool]:
    return lambda f: f.get(key) is not None and f[key] > limit


def _lt(key: str, limit: float) -> Callable[[Features], bool]:
    return lambda f: f.get(key) is not None and f[key] < limit


DEFAULT_RULES: tuple[RiskRule, ...] = (
    RiskRule("high_dti", _gt("debt_to_income", 0.43), "Debt-to-income ratio is above 43%."),
    RiskRule(
        "high_utilization",
        _gt("credit_utilization", FEATURE_SETTINGS.high_utilization_threshold),
        "Credit utilisation is at or above 70%.",
    ),
    RiskRule("large_loan", _gt("loan_to_income", 1.0), "Requested loan exceeds annual income."),
    RiskRule("repeated_late_payments", _gt("num_late_payments", 2), "Three or more late payments on record."),
    RiskRule("thin_savings", _lt("savings_runway_months", 3.0), "Savings cover less than three months of expenses."),
)


class RecommendationEngine:
    """Combine a segment policy with optional consideration rules."""

    def __init__(
        self,
        policy: Mapping[str, tuple[str, str, str]] = DEFAULT_POLICY,
        rules: Sequence[RiskRule] = DEFAULT_RULES,
    ) -> None:
        self.policy = dict(policy)
        self.rules = tuple(rules)
        self._by_action = {action: (title, message) for action, title, message in self.policy.values()}

    def considerations(self, features: Features | None) -> list[str]:
        """Messages of every rule that fires for ``features``."""
        if not features:
            return []
        return [rule.message for rule in self.rules if rule.applies(features)]

    def recommend(self, risk_segment: str, features: Features | None = None) -> Recommendation:
        """Recommendation for a segment code (``low`` / ``medium`` / ``high``)."""
        action, title, message = self.policy[risk_segment]
        return Recommendation(action, title, message, self.considerations(features))

    def from_stored(self, action: str, considerations: list[str] | None) -> Recommendation:
        """Rebuild a recommendation from the values saved with a prediction."""
        title, message = self._by_action.get(action, (action.replace("_", " ").title(), ""))
        return Recommendation(action, title, message, list(considerations or []))


def derive_features(record: Mapping) -> dict[str, float | None]:
    """Engineered ratios for one raw record, reusing the ML feature code (no re-implementation)."""
    frame = engineer_features(clean_for_inference(pd.DataFrame([dict(record)])))
    row = frame.iloc[0]
    return {
        key: (None if pd.isna(value) else float(value))
        for key, value in row.items()
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
    }
