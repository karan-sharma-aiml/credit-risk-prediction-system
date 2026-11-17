"""Reusable customer input form."""

from __future__ import annotations

from typing import Any

import streamlit as st

EMPLOYMENT = ["salaried", "self_employed", "retired", "unemployed"]
REPAYMENT = ["excellent", "good", "fair", "poor"]
HOME = ["rent", "mortgage", "own", "other"]
NONE_LABEL = "(unknown)"

DEFAULTS: dict[str, Any] = {
    "age": 35, "annual_income": 60000.0, "employment_type": "salaried", "employment_years": 5.0,
    "loan_amount": 15000.0, "loan_term_months": 36, "credit_score": 680, "monthly_debt_payment": 600.0,
    "credit_limit": 12000.0, "revolving_balance": 3000.0, "savings_balance": 8000.0,
    "monthly_expenses": 2500.0, "num_existing_loans": 1, "num_late_payments": 0,
    "repayment_history": "good", "home_ownership": "rent",
}


def _select_optional(label: str, options: list[str], current: str | None, key: str) -> str | None:
    choices = [NONE_LABEL, *options]
    index = choices.index(current) if current in choices else 0
    value = st.selectbox(label, choices, index=index, key=key)
    return None if value == NONE_LABEL else value


def customer_form(
    key: str,
    defaults: dict[str, Any] | None = None,
    *,
    submit_label: str = "Submit",
    include_outcome: bool = False,
) -> dict[str, Any] | None:
    """Render the customer form; return the payload when submitted, else ``None``.

    Optional model inputs (credit limit, balances, history, ...) may be left unknown.
    """
    d = {**DEFAULTS, **{k: v for k, v in (defaults or {}).items() if v is not None}}
    with st.form(key):
        left, right = st.columns(2)
        with left:
            age = st.number_input("Age", 18, 100, int(d["age"]), key=f"{key}_age")
            income = st.number_input("Annual income", 1.0, 10_000_000.0, float(d["annual_income"]), 1000.0, key=f"{key}_inc")
            employment = st.selectbox("Employment type", EMPLOYMENT, EMPLOYMENT.index(d["employment_type"]), key=f"{key}_emp")
            emp_years = st.number_input("Years in current employment", 0.0, 60.0, float(d["employment_years"]), 0.5, key=f"{key}_empy")
            loan = st.number_input("Loan amount", 1.0, 10_000_000.0, float(d["loan_amount"]), 500.0, key=f"{key}_loan")
            term = st.number_input("Loan term (months)", 1, 480, int(d["loan_term_months"]), key=f"{key}_term")
            score = st.number_input("Credit score", 300, 850, int(d["credit_score"]), key=f"{key}_score")
            debt = st.number_input("Monthly debt payments", 0.0, 1_000_000.0, float(d["monthly_debt_payment"]), 50.0, key=f"{key}_debt")
        with right:
            limit = st.number_input("Credit limit", 0.0, 100_000_000.0, float(d["credit_limit"]), 500.0, key=f"{key}_limit")
            revolving = st.number_input("Revolving balance", 0.0, 100_000_000.0, float(d["revolving_balance"]), 100.0, key=f"{key}_rev")
            savings = st.number_input("Savings balance", 0.0, 100_000_000.0, float(d["savings_balance"]), 500.0, key=f"{key}_sav")
            expenses = st.number_input("Monthly expenses", 0.0, 1_000_000.0, float(d["monthly_expenses"]), 50.0, key=f"{key}_exp")
            loans = st.number_input("Existing loans", 0, 50, int(d["num_existing_loans"]), key=f"{key}_loans")
            late = st.number_input("Late payments", 0, 100, int(d["num_late_payments"]), key=f"{key}_late")
            history = _select_optional("Repayment history", REPAYMENT, d["repayment_history"], f"{key}_hist")
            home = _select_optional("Home ownership", HOME, d["home_ownership"], f"{key}_home")
            outcome = None
            if include_outcome:
                choice = st.selectbox("Observed outcome", [NONE_LABEL, "repaid", "defaulted"], key=f"{key}_out")
                outcome = {NONE_LABEL: None, "repaid": False, "defaulted": True}[choice]
        submitted = st.form_submit_button(submit_label, type="primary")

    if not submitted:
        return None
    payload: dict[str, Any] = {
        "age": int(age), "annual_income": float(income), "employment_type": employment,
        "employment_years": float(emp_years), "loan_amount": float(loan), "loan_term_months": int(term),
        "credit_score": int(score), "monthly_debt_payment": float(debt), "credit_limit": float(limit),
        "revolving_balance": float(revolving), "savings_balance": float(savings),
        "monthly_expenses": float(expenses), "num_existing_loans": int(loans), "num_late_payments": int(late),
        "repayment_history": history, "home_ownership": home,
    }
    if include_outcome:
        payload["is_default"] = outcome
    return payload
