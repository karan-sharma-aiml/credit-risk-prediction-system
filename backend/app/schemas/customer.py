"""Customer / prediction-input schemas with business validation (Pydantic v2).

The ``Literal`` categories duplicate ``ml.config.CATEGORY_LEVELS`` on purpose
(explicit OpenAPI enums); ``tests/backend/test_schemas.py`` fails if they drift.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

EmploymentType = Literal["salaried", "self_employed", "retired", "unemployed"]
RepaymentHistory = Literal["excellent", "good", "fair", "poor"]
HomeOwnership = Literal["rent", "mortgage", "own", "other"]

CUSTOMER_ID_PATTERN = r"^[A-Za-z0-9_-]{3,20}$"
MIN_WORKING_AGE = 14


class CustomerBase(BaseModel):
    """Raw customer features (same schema the ML model was trained on)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    age: int = Field(ge=18, le=100, description="Age in years")
    annual_income: float = Field(gt=0, le=10_000_000, description="Gross annual income")
    employment_type: EmploymentType
    employment_years: float | None = Field(default=None, ge=0, le=60)
    loan_amount: float = Field(gt=0, le=10_000_000, description="Requested principal")
    loan_term_months: int = Field(ge=1, le=480)
    credit_score: int = Field(ge=300, le=850)
    monthly_debt_payment: float = Field(ge=0, le=1_000_000, description="Monthly payments on existing debt")
    credit_limit: float | None = Field(default=None, ge=0)
    revolving_balance: float | None = Field(default=None, ge=0)
    savings_balance: float = Field(ge=0)
    monthly_expenses: float = Field(ge=0, le=1_000_000, description="Monthly living expenses")
    num_existing_loans: int | None = Field(default=None, ge=0, le=50)
    num_late_payments: int | None = Field(default=None, ge=0, le=100)
    repayment_history: RepaymentHistory | None = None
    home_ownership: HomeOwnership | None = None

    @model_validator(mode="after")
    def _tenure_fits_age(self) -> "CustomerBase":
        if self.employment_years is not None and self.employment_years > self.age - MIN_WORKING_AGE:
            raise ValueError("employment_years cannot exceed (age - 14).")
        return self


class CustomerCreate(CustomerBase):
    """Body of ``POST /customers``."""

    customer_id: str | None = Field(default=None, pattern=CUSTOMER_ID_PATTERN)
    is_default: bool | None = Field(default=None, description="Observed outcome, if known")
    source_dataset: str = Field(default="api", max_length=100)


class CustomerUpdate(CustomerBase):
    """Body of ``PUT /customers/{id}`` (full replacement; omitted optional fields become null)."""

    is_default: bool | None = None


class CustomerRead(BaseModel):
    """Customer as stored. Deliberately unconstrained: it must be able to represent any stored row."""

    model_config = ConfigDict(from_attributes=True)

    customer_id: str
    age: int | None = None
    annual_income: float | None = None
    employment_type: str | None = None
    employment_years: float | None = None
    loan_amount: float | None = None
    loan_term_months: int | None = None
    credit_score: int | None = None
    monthly_debt_payment: float | None = None
    credit_limit: float | None = None
    revolving_balance: float | None = None
    savings_balance: float | None = None
    monthly_expenses: float | None = None
    num_existing_loans: int | None = None
    num_late_payments: int | None = None
    repayment_history: str | None = None
    home_ownership: str | None = None
    is_default: bool | None = None
    source_dataset: str
    created_at: datetime
    updated_at: datetime
