"""Plotly figure builders (pure functions: data in, figure out)."""

from __future__ import annotations

from collections.abc import Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.config import RISK_COLORS

_LAYOUT = {"margin": {"l": 10, "r": 10, "t": 50, "b": 10}, "template": "plotly_white", "height": 360}


def risk_distribution_chart(buckets: Sequence[dict]) -> go.Figure:
    """Donut of customers per current risk segment."""
    fig = go.Figure(
        go.Pie(
            labels=[b["label"] for b in buckets],
            values=[b["count"] for b in buckets],
            hole=0.55,
            marker={"colors": [RISK_COLORS.get(b["segment"], "#888") for b in buckets]},
            sort=False,
        )
    )
    return fig.update_layout(title="Risk distribution (latest prediction per customer)", **_LAYOUT)


def histogram_chart(hist: dict, title: str, x_title: str, color: str = "#4C78A8") -> go.Figure:
    """Bar histogram from ``{"bin_edges": [...], "counts": [...]}``."""
    edges, counts = hist.get("bin_edges", []), hist.get("counts", [])
    fig = go.Figure()
    if counts:
        mids = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
        width = [b - a for a, b in zip(edges[:-1], edges[1:])]
        fig.add_bar(x=mids, y=counts, width=width, marker_color=color)
    return fig.update_layout(title=title, xaxis_title=x_title, yaxis_title="customers", bargap=0.05, **_LAYOUT)


def trend_chart(points: Sequence[dict]) -> go.Figure:
    """Daily prediction volume (bars) and mean default probability (line)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    dates = [p["date"] for p in points]
    fig.add_bar(x=dates, y=[p["count"] for p in points], name="predictions", marker_color="#9ecae1")
    fig.add_scatter(
        x=dates, y=[p["avg_probability"] for p in points], name="avg default probability",
        mode="lines+markers", line={"color": "#D64545"}, secondary_y=True,
    )
    fig.update_yaxes(title_text="predictions", secondary_y=False)
    fig.update_yaxes(title_text="avg probability", tickformat=".0%", secondary_y=True)
    return fig.update_layout(title="Prediction trend (last 30 days)", **_LAYOUT)


def feature_importance_chart(items: Sequence[dict]) -> go.Figure:
    """Horizontal bars of the most important features."""
    ordered = sorted(items, key=lambda i: i["importance"])
    fig = go.Figure(go.Bar(x=[i["importance"] for i in ordered], y=[i["feature"] for i in ordered],
                           orientation="h", marker_color="#4C78A8"))
    return fig.update_layout(title="Feature importance (top features)", **{**_LAYOUT, "height": 460})


def curve_chart(curve: dict, title: str, x_title: str, y_title: str, baseline: bool = False) -> go.Figure:
    """ROC or precision-recall curve with its area in the legend."""
    fig = go.Figure(go.Scatter(x=curve["x"], y=curve["y"], mode="lines", name=f"AUC / AP = {curve['auc']:.3f}",
                               line={"color": "#4C78A8", "width": 3}))
    if baseline:
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash", "color": "grey"})
    return fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title,
                             xaxis={"range": [0, 1]}, yaxis={"range": [0, 1.02]}, **_LAYOUT)


def confusion_matrix_chart(cm: dict) -> go.Figure:
    """Annotated 2x2 heatmap (rows = actual, columns = predicted)."""
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    fig = go.Figure(go.Heatmap(z=z, x=["Predicted repaid", "Predicted default"], y=["Actual repaid", "Actual default"],
                               text=[[f"{v:,}" for v in row] for row in z], texttemplate="%{text}",
                               colorscale="Blues", showscale=False))
    return fig.update_layout(title="Confusion matrix (test set)", yaxis={"autorange": "reversed"}, **_LAYOUT)


def probability_gauge(probability: float, low_max: float, high_min: float) -> go.Figure:
    """Gauge of one customer's default probability with the segment bands."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#333"},
                "steps": [
                    {"range": [0, low_max * 100], "color": RISK_COLORS["low"]},
                    {"range": [low_max * 100, high_min * 100], "color": RISK_COLORS["medium"]},
                    {"range": [high_min * 100, 100], "color": RISK_COLORS["high"]},
                ],
            },
            title={"text": "Default probability"},
        )
    )
    return fig.update_layout(margin={"l": 20, "r": 20, "t": 60, "b": 10}, height=300)
