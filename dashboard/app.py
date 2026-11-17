"""Streamlit entry point:  streamlit run dashboard/app.py"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `dashboard` importable

import streamlit as st  # noqa: E402

from dashboard.api_client import APIError, get_client  # noqa: E402
from dashboard.views import analytics, customers, dashboard, history, prediction  # noqa: E402

PAGES = {
    "Dashboard": dashboard,
    "Customer Management": customers,
    "Prediction": prediction,
    "Analytics": analytics,
    "Prediction History": history,
}


def main() -> None:
    st.set_page_config(page_title="Credit Risk Platform", page_icon="💳", layout="wide")
    client = get_client()

    st.sidebar.title("💳 Credit Risk")
    page = st.sidebar.radio("Navigate", list(PAGES), key="page")
    try:
        health = client.health()
        st.sidebar.success(f"API online · {health['model']}")
    except APIError as exc:
        st.sidebar.error("API unavailable")
        st.error(exc.message)
        st.info("Start the backend (`uvicorn backend.app.main:app_factory --factory`) or set `API_URL`.")
        return

    try:
        PAGES[page].render(client)
    except APIError as exc:
        st.error(exc.message)


main()
