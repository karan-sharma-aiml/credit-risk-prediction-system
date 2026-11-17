"""Prediction page: score a new or existing customer."""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import APIClient, APIError
from dashboard.charts import probability_gauge
from dashboard.forms import customer_form
from dashboard.ui import risk_badge, show_api_error

CUSTOMER_FIELDS = (
    "age", "annual_income", "employment_type", "employment_years", "loan_amount", "loan_term_months",
    "credit_score", "monthly_debt_payment", "credit_limit", "revolving_balance", "savings_balance",
    "monthly_expenses", "num_existing_loans", "num_late_payments", "repayment_history", "home_ownership",
)


def _show_result(result: dict, info: dict) -> None:
    cut = info["segment_cutoffs"]
    st.divider()
    left, right = st.columns([1, 1])
    left.plotly_chart(probability_gauge(result["default_probability"], cut["low_max"], cut["high_min"]),
                      width="stretch")
    with right:
        st.markdown(f"### {risk_badge(result['risk_segment'], result['risk_label'])}")
        st.metric("Predicted class", "Default" if result["predicted_label"] else "Repaid")
        rec = result["recommendation"]
        st.markdown(f"**Recommendation: {rec['title']}**")
        st.write(rec["message"])
        for note in rec["considerations"]:
            st.warning(note)
        st.caption(f"Customer {result['customer_id']} · prediction #{result['prediction_id']} · {result['model_version']}")
    history = result.get("history") or {}
    if history.get("previous_segment"):
        delta = history["probability_change"]
        st.info(f"Previous prediction: {history['previous_segment']} risk "
                f"({delta:+.1%} change in default probability).")


def render(client: APIClient) -> None:
    st.title("Prediction")
    info = client.model_info()
    mode = st.radio("Customer", ["New customer", "Existing customer"], horizontal=True, key="pred_mode")

    defaults: dict | None = None
    customer_id: str | None = None
    if mode == "Existing customer":
        customer_id = st.text_input("Customer id", key="pred_customer_id").strip() or None
        if not customer_id:
            st.info("Enter a customer id to load their data.")
            return
        try:
            defaults = client.get_customer(customer_id)
        except APIError as exc:
            show_api_error(exc)
            return

    payload = customer_form(f"predict_{customer_id or 'new'}", defaults, submit_label="Predict default risk")
    if payload is None:
        return
    if customer_id:
        payload["customer_id"] = customer_id
    try:
        result = client.predict(payload)
    except APIError as exc:
        show_api_error(exc)
        return
    _show_result(result, info)
