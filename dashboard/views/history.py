"""Prediction history page: filter, inspect and export."""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import APIClient, APIError
from dashboard.config import RISK_LABELS
from dashboard.ui import predictions_frame, show_api_error

PAGE_SIZE = 50


def render(client: APIClient) -> None:
    st.title("Prediction History")
    c1, c2, c3 = st.columns([2, 1, 1])
    customer_id = c1.text_input("Customer id", key="hist_customer").strip() or None
    segment = c2.selectbox("Risk segment", ["all", *RISK_LABELS], key="hist_segment")
    page_no = c3.number_input("Page", min_value=1, value=1, key="hist_page")

    data = client.list_predictions(
        limit=PAGE_SIZE, offset=(int(page_no) - 1) * PAGE_SIZE,
        customer_id=customer_id, risk_segment=None if segment == "all" else segment,
    )
    st.caption(f"{data['total']:,} predictions")
    if not data["items"]:
        st.info("No predictions match the filters.")
        return

    frame = predictions_frame(data["items"])
    st.dataframe(frame, width="stretch", hide_index=True)
    st.download_button("Download CSV", frame.to_csv(index=False), "prediction_history.csv", "text/csv")

    chosen = st.selectbox("Inspect prediction", frame["prediction_id"].tolist(), key="hist_inspect")
    try:
        detail = client.get_prediction(int(chosen))
    except APIError as exc:
        show_api_error(exc)
        return
    history = detail.get("history") or {}
    left, right = st.columns(2)
    left.subheader("Model input (audit snapshot)")
    left.json(history.get("input_snapshot", {}))
    right.subheader("Outcome")
    right.write({
        "probability": detail["default_probability"], "risk": detail["risk_label"],
        "recommendation": detail["recommendation"]["title"], "considerations": detail["recommendation"]["considerations"],
        "previous_segment": history.get("previous_segment"), "segment_changed": history.get("segment_changed"),
        "probability_change": history.get("probability_change"), "request_id": detail.get("request_id"),
    })
