"""Dashboard page: KPIs, risk distribution, trend, recent predictions."""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import APIClient
from dashboard.charts import risk_distribution_chart, trend_chart
from dashboard.ui import predictions_frame


def render(client: APIClient) -> None:
    st.title("Dashboard")
    data = client.dashboard()
    kpis = data["kpis"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{kpis['total_customers']:,}")
    c2.metric("Predictions", f"{kpis['total_predictions']:,}")
    avg = kpis["average_default_probability"]
    c3.metric("Avg default probability", "—" if avg is None else f"{avg:.1%}")
    share = kpis["high_risk_share"]
    c4.metric("High-risk customers", "—" if share is None else f"{share:.1%}")

    if kpis["total_predictions"] == 0:
        st.info("No predictions yet. Score a customer on the **Prediction** page, "
                "or load demo data with `python -m backend.app.scripts.seed_demo`.")
        return

    left, right = st.columns(2)
    left.plotly_chart(risk_distribution_chart(data["risk_distribution"]), width="stretch")
    right.plotly_chart(trend_chart(data["prediction_trend"]), width="stretch")

    st.subheader("Recent predictions")
    st.dataframe(
        predictions_frame(data["recent_predictions"]),
        width="stretch", hide_index=True,
        column_config={"probability": st.column_config.ProgressColumn("probability", format="%.3f", min_value=0, max_value=1)},
    )
