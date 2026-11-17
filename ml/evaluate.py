"""Model evaluation: metrics, threshold selection and report plots."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ml.config import SELECTION_SETTINGS, SelectionSettings  # noqa: E402
from ml.eda import save_figure  # noqa: E402

logger = logging.getLogger(__name__)

METRIC_NAMES: tuple[str, ...] = (
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score",
)
_PLOT_COLORS = plt.get_cmap("tab10")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def select_threshold(
    y_true: np.ndarray | pd.Series,
    probabilities: np.ndarray,
    settings: SelectionSettings = SELECTION_SETTINGS,
) -> float:
    """Choose the decision threshold on (validation) data.

    Maximises F-beta (recall-weighted) among thresholds whose precision is at
    least ``settings.min_precision``; falls back to the F1-optimal threshold if
    none qualifies. A customer is flagged as default when ``p >= threshold``.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    p, r = precision[:-1], recall[:-1]

    def fbeta(beta: float) -> np.ndarray:
        b2 = beta**2
        denom = b2 * p + r
        return np.divide((1 + b2) * p * r, denom, out=np.zeros_like(denom), where=denom > 0)

    eligible = p >= settings.min_precision
    if eligible.any():
        scores = np.where(eligible, fbeta(settings.threshold_beta), -1.0)
    else:
        logger.warning("No threshold reaches min precision %.2f; using F1-optimal", settings.min_precision)
        scores = fbeta(1.0)
    return float(np.clip(thresholds[int(np.argmax(scores))], 1e-6, 1 - 1e-6))


def compute_metrics(
    y_true: np.ndarray | pd.Series, probabilities: np.ndarray, threshold: float
) -> dict:
    """Classification metrics at ``threshold`` plus threshold-free ranking metrics.

    Returns a JSON-serialisable dict containing accuracy, precision, recall, f1,
    roc_auc, pr_auc, brier_score, calibration summary and the confusion matrix
    (``tn``/``fp``/``fn``/``tp``).
    """
    y = np.asarray(y_true).astype(int)
    pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "pr_auc": float(average_precision_score(y, probabilities)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "mean_predicted_probability": float(np.mean(probabilities)),
        "observed_default_rate": float(y.mean()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


@dataclass
class ModelEvaluation:
    """Evaluation of one model on validation and test data."""

    model_name: str
    threshold: float
    validation: dict
    test: dict
    probabilities: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        """JSON-friendly view (probabilities excluded)."""
        return {
            "model_name": self.model_name,
            "threshold": self.threshold,
            "validation": self.validation,
            "test": self.test,
        }


def evaluate_probabilities(
    model_name: str,
    y_val: pd.Series,
    proba_val: np.ndarray,
    y_test: pd.Series,
    proba_test: np.ndarray,
    settings: SelectionSettings = SELECTION_SETTINGS,
) -> ModelEvaluation:
    """Pick the threshold on validation, then score validation and test with it.

    The test set never influences the threshold.
    """
    threshold = select_threshold(y_val, proba_val, settings)
    return ModelEvaluation(
        model_name=model_name,
        threshold=threshold,
        validation=compute_metrics(y_val, proba_val, threshold),
        test=compute_metrics(y_test, proba_test, threshold),
        probabilities={"validation": proba_val, "test": proba_test},
    )


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def comparison_table(evaluations: Sequence[ModelEvaluation]) -> pd.DataFrame:
    """Wide table: one row per model, ``val_*`` and ``test_*`` metric columns."""
    rows = []
    for ev in evaluations:
        row: dict = {"model": ev.model_name, "threshold": round(ev.threshold, 4)}
        for split, metrics in (("val", ev.validation), ("test", ev.test)):
            for name in METRIC_NAMES:
                row[f"{split}_{name}"] = round(metrics[name], 4)
        rows.append(row)
    return pd.DataFrame(rows)


def metrics_long_table(evaluations: Sequence[ModelEvaluation]) -> pd.DataFrame:
    """Long format matching ``sql/schema/004_model_metrics.sql``.

    Columns: ``model_name, dataset_split, metric_name, metric_value``.
    """
    rows = []
    for ev in evaluations:
        for split, metrics in (("validation", ev.validation), ("test", ev.test)):
            scalar = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
            scalar.update(metrics["confusion_matrix"])
            rows.extend(
                {"model_name": ev.model_name, "dataset_split": split,
                 "metric_name": k, "metric_value": round(float(v), 6)}
                for k, v in scalar.items()
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plots (all computed on the held-out test split)
# --------------------------------------------------------------------------- #
def plot_roc_curves(
    evaluations: Sequence[ModelEvaluation], y_test: pd.Series, path: Path, best: str | None = None
) -> Path:
    """Overlay ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i, ev in enumerate(evaluations):
        fpr, tpr, _ = roc_curve(y_test, ev.probabilities["test"])
        width = 2.8 if ev.model_name == best else 1.4
        ax.plot(fpr, tpr, color=_PLOT_COLORS(i), lw=width,
                label=f"{ev.model_name} (AUC {ev.test['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (test set)")
    ax.legend(fontsize=8, loc="lower right")
    return save_figure(fig, path)


def plot_precision_recall_curves(
    evaluations: Sequence[ModelEvaluation], y_test: pd.Series, path: Path, best: str | None = None
) -> Path:
    """Overlay precision-recall curves, marking each model's operating point."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i, ev in enumerate(evaluations):
        precision, recall, _ = precision_recall_curve(y_test, ev.probabilities["test"])
        width = 2.8 if ev.model_name == best else 1.4
        ax.plot(recall, precision, color=_PLOT_COLORS(i), lw=width,
                label=f"{ev.model_name} (AP {ev.test['pr_auc']:.3f})")
        ax.scatter([ev.test["recall"]], [ev.test["precision"]], color=_PLOT_COLORS(i), s=35, zorder=3)
    ax.axhline(float(np.mean(y_test)), color="k", linestyle="--", lw=1, label="baseline (default rate)")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (test set; dots = chosen threshold)")
    ax.legend(fontsize=8, loc="upper right")
    return save_figure(fig, path)


def plot_confusion_matrices(evaluations: Sequence[ModelEvaluation], path: Path) -> Path:
    """Grid of test-set confusion matrices at each model's threshold."""
    n = len(evaluations)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows), squeeze=False)
    for ax, ev in zip(axes.ravel(), evaluations):
        cm = ev.test["confusion_matrix"]
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f"{value:,}", ha="center", va="center",
                    color="white" if value > matrix.max() / 2 else "black")
        ax.set_xticks([0, 1], ["pred repaid", "pred default"], fontsize=8)
        ax.set_yticks([0, 1], ["actual repaid", "actual default"], fontsize=8)
        ax.set_title(f"{ev.model_name}\nthreshold {ev.threshold:.2f}", fontsize=10)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Confusion matrices (test set)", y=1.0)
    return save_figure(fig, path)


def plot_probability_distribution(
    probabilities: np.ndarray, y_true: pd.Series, threshold: float, path: Path, title: str
) -> Path:
    """Histogram of predicted default probabilities split by actual outcome."""
    y = np.asarray(y_true).astype(int)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 41)
    ax.hist(probabilities[y == 0], bins=bins, alpha=0.7, color="#4C78A8", label="actual repaid")
    ax.hist(probabilities[y == 1], bins=bins, alpha=0.7, color="#E45756", label="actual default")
    ax.axvline(threshold, color="black", linestyle="--", label=f"decision threshold {threshold:.2f}")
    ax.set(xlabel="Predicted default probability", ylabel="customers", title=title)
    ax.legend()
    return save_figure(fig, path)


def plot_model_comparison(table: pd.DataFrame, path: Path, metrics: Mapping[str, str] | None = None) -> Path:
    """Grouped bar chart of key validation metrics per model."""
    metrics = metrics or {"val_recall": "recall", "val_f1": "F1", "val_roc_auc": "ROC-AUC", "val_pr_auc": "PR-AUC"}
    x = np.arange(len(table))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(table)), 4.5))
    for k, (col, label) in enumerate(metrics.items()):
        ax.bar(x + k * width, table[col], width, label=label)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2, table["model"], rotation=20, ha="right")
    ax.set(ylim=(0, 1), title="Model comparison (validation set)")
    ax.legend(ncol=len(metrics), fontsize=8)
    return save_figure(fig, path)
