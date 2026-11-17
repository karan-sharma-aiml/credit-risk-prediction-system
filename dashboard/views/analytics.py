"""Analytics page: portfolio distributions and model diagnostics."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.api_client import APIClient
from dashboard.charts import (
    confusion_matrix_chart,
    curve_chart,
    feature_importance_chart,
    histogram_chart,
)


def _portfolio(data: dict) -> None:
    if not data["income_distribution"]["counts"]:
        st.info("No customers yet.")
        return
    left, right = st.columns(2)
    left.plotly_chart(histogram_chart(data["income_distribution"], "Income distribution", "annual income"),
                      width="stretch")
    right.plotly_chart(histogram_chart(data["loan_distribution"], "Loan amount distribution", "loan amount", "#F58518"),
                       width="stretch")
    st.plotly_chart(histogram_chart(data["credit_score_distribution"], "Credit score distribution", "credit score", "#54A24B"),
                    width="stretch")


def _model(client: APIClient) -> None:
    info, perf = client.model_info(), client.model_performance()
    st.caption(f"Model **{info['model_version']}** · decision threshold {info['decision_threshold']:.3f} · "
               f"SMOTE: {info.get('balanced_with_smote')}")
    metrics = pd.DataFrame({"validation": info["validation_metrics"], "test": info["test_metrics"]}).round(4)
    st.dataframe(metrics, width="stretch")

    if perf["feature_importance"]:
        st.plotly_chart(feature_importance_chart(perf["feature_importance"]), width="stretch")
    left, right = st.columns(2)
    if perf["roc_curve"]:
        left.plotly_chart(curve_chart(perf["roc_curve"], "ROC curve (test set)", "False positive rate",
                                      "True positive rate", baseline=True), width="stretch")
    if perf["pr_curve"]:
        right.plotly_chart(curve_chart(perf["pr_curve"], "Precision-recall curve (test set)", "Recall", "Precision"),
                           width="stretch")
    if perf["confusion_matrix"]:
        st.plotly_chart(confusion_matrix_chart(perf["confusion_matrix"]), width="stretch")


def render(client: APIClient) -> None:
    st.title("Analytics")
    portfolio, model = st.tabs(["Portfolio", "Model performance"])
    with portfolio:
        _portfolio(client.dashboard())
    with model:
        _model(client)
