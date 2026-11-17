"""Small shared Streamlit helpers."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.api_client import APIError
from dashboard.config import RISK_COLORS


def show_api_error(exc: APIError) -> None:
    """Render an API error with per-field validation messages."""
    st.error(exc.message)
    for detail in exc.details:
        st.caption(f"• **{detail.get('field') or 'request'}**: {detail.get('message')}")


def risk_badge(segment: str, label: str) -> str:
    """Coloured markdown badge for a risk segment."""
    return f":{ {'low': 'green', 'medium': 'orange', 'high': 'red'}.get(segment, 'gray') }-badge[{label}]"


def predictions_frame(items: list[dict]) -> pd.DataFrame:
    """Flatten prediction records for display."""
    rows = [
        {
            "prediction_id": p["prediction_id"],
            "customer_id": p["customer_id"],
            "probability": p["default_probability"],
            "risk": p["risk_label"],
            "label": "default" if p["predicted_label"] else "repaid",
            "recommendation": p["recommendation"]["title"],
            "model_version": p["model_version"],
            "created_at": p["created_at"],
        }
        for p in items
    ]
    return pd.DataFrame(rows)


def risk_color(segment: str) -> str:
    return RISK_COLORS.get(segment, "#888")
