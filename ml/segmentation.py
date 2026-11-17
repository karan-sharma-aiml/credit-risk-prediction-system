"""Convert default probabilities into business risk segments.

Thresholds come exclusively from ``ml.config.SEGMENT_SETTINGS``; no other
module hardcodes a cut-off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.config import SEGMENT_SETTINGS, SegmentSettings


def validate_segment_settings(settings: SegmentSettings) -> None:
    """Raise ``ValueError`` unless ``0 < low_max < high_min < 1``."""
    if not 0 < settings.low_max < settings.high_min < 1:
        raise ValueError("Segment thresholds must satisfy 0 < low_max < high_min < 1.")
    if len(settings.codes) != 3 or len(settings.labels) != 3:
        raise ValueError("Exactly three segment codes and labels are required.")


def assign_risk_segment(
    probabilities: np.ndarray | pd.Series,
    settings: SegmentSettings = SEGMENT_SETTINGS,
    flag_threshold: float | None = None,
) -> pd.DataFrame:
    """Map probabilities to segments.

    Args:
        probabilities: Default probabilities in ``[0, 1]``.
        settings: Cut-offs and names.
        flag_threshold: The model's decision threshold. When given, any customer
            the model flags as a likely defaulter (``p >= flag_threshold``) is
            at least "medium", so a flagged customer is never shown as
            "Low Risk".

    Returns:
        DataFrame with ``risk_segment`` (code: low/medium/high, matches the SQL
        ``predictions`` table) and ``risk_label`` (display name).
    """
    validate_segment_settings(settings)
    p = np.asarray(probabilities, dtype=float)
    if np.isnan(p).any() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1].")
    index = probabilities.index if isinstance(probabilities, pd.Series) else None
    low_max = settings.low_max if flag_threshold is None else min(settings.low_max, flag_threshold)
    code = np.select([p < low_max, p < settings.high_min], settings.codes[:2], settings.codes[2])
    label = pd.Series(code).map(dict(zip(settings.codes, settings.labels))).to_numpy()
    return pd.DataFrame({"risk_segment": code, "risk_label": label}, index=index)


def summarize_segments(
    probabilities: np.ndarray | pd.Series,
    y_true: np.ndarray | pd.Series | None = None,
    settings: SegmentSettings = SEGMENT_SETTINGS,
    flag_threshold: float | None = None,
) -> pd.DataFrame:
    """Per-segment size, mean probability and (if labels are given) observed default rate.

    A well-behaved segmentation shows a monotonic default rate: low < medium < high.
    """
    seg = assign_risk_segment(probabilities, settings, flag_threshold).reset_index(drop=True)
    frame = pd.DataFrame({"probability": np.asarray(probabilities, dtype=float)})
    frame["risk_segment"] = seg["risk_segment"].to_numpy()
    if y_true is not None:
        frame["actual_default"] = np.asarray(y_true)
    grouped = frame.groupby("risk_segment")
    out = pd.DataFrame({"n_customers": grouped.size(), "mean_probability": grouped["probability"].mean()})
    if y_true is not None:
        out["observed_default_rate"] = grouped["actual_default"].mean()
    out = out.reindex(list(settings.codes)).fillna({"n_customers": 0})
    out["n_customers"] = out["n_customers"].astype(int)
    out["share"] = (out["n_customers"] / max(len(frame), 1)).round(4)
    out.insert(0, "risk_label", list(settings.labels))
    return out.round(4).reset_index(names="risk_segment")
