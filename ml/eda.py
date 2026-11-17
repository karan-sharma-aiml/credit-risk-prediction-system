"""Exploratory data analysis: summary tables and saved plots.

Every function is pure (takes a DataFrame, returns a table or writes one
figure), so the same code backs the pipeline, notebooks and tests.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml.config import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    CATEGORY_LEVELS,
    ENGINEERED_CATEGORICAL_COLUMNS,
    FEATURE_SETTINGS,
    ENGINEERED_NUMERIC_COLUMNS,
    NUMERIC_COLUMNS,
    PATHS,
    TARGET_COL,
)
from utils.io_utils import ensure_dir, save_json  # noqa: E402

logger = logging.getLogger(__name__)

KEY_RISK_VARIABLES: tuple[str, ...] = (
    "credit_score",
    "debt_to_income",
    "loan_to_income",
    "credit_utilization",
    "num_late_payments",
    "savings_ratio",
)
_COLORS = {0: "#4C78A8", 1: "#E45756"}


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def class_distribution(df: pd.DataFrame, target: str = TARGET_COL) -> pd.DataFrame:
    """Counts and percentages of each target class."""
    counts = df[target].value_counts().sort_index()
    return pd.DataFrame({"count": counts, "pct": (counts / counts.sum() * 100).round(2)})


def numeric_summary(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """``describe()`` plus skewness for the given numeric columns."""
    cols = [c for c in columns if c in df.columns]
    summary = df[cols].describe().T
    summary["skew"] = df[cols].skew()
    return summary.round(4)


def correlation_matrix(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Pearson correlation matrix over the given numeric columns."""
    cols = [c for c in columns if c in df.columns]
    return df[cols].corr()


def target_correlations(
    df: pd.DataFrame, columns: Sequence[str], target: str = TARGET_COL
) -> pd.Series:
    """Correlation of each numeric feature with the target, sorted by magnitude."""
    cols = [c for c in columns if c in df.columns and c != target]
    corr = df[cols].corrwith(df[target])
    return corr.reindex(corr.abs().sort_values(ascending=False).index).round(4)


def _level_order(column: str) -> list[str] | None:
    """Natural (not alphabetical) level order for known categorical columns."""
    known = {
        **{k: list(v) for k, v in CATEGORY_LEVELS.items()},
        "age_group": list(FEATURE_SETTINGS.age_labels),
        "income_bucket": list(FEATURE_SETTINGS.income_labels),
    }
    return known.get(column)


def default_rate_by_category(
    df: pd.DataFrame, column: str, target: str = TARGET_COL
) -> pd.DataFrame:
    """Default rate and volume per level of a categorical column."""
    grouped = df.groupby(column, dropna=False)[target].agg(["count", "mean"])
    order = _level_order(column)
    if order:
        present = [lvl for lvl in order if lvl in grouped.index]
        grouped = pd.concat([grouped.loc[present], grouped.drop(index=present)])
    grouped.columns = ["count", "default_rate"]
    grouped.insert(0, "feature", column)
    return grouped.reset_index().rename(columns={column: "level"}).round(4)


def compute_feature_summary(
    df: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> pd.DataFrame:
    """Per-feature statistics. Column names match ``sql/schema/005_feature_summaries.sql``."""
    rows: list[dict] = []
    for col in numeric_columns:
        if col not in df.columns:
            continue
        s = df[col]
        rows.append({
            "feature_name": col, "feature_type": "numeric", "n_obs": int(s.notna().sum()),
            "n_missing": int(s.isna().sum()), "mean": s.mean(), "std": s.std(),
            "min_value": s.min(), "p25": s.quantile(0.25), "median": s.median(),
            "p75": s.quantile(0.75), "max_value": s.max(), "n_unique": int(s.nunique()),
        })
    for col in categorical_columns:
        if col not in df.columns:
            continue
        s = df[col]
        rows.append({
            "feature_name": col, "feature_type": "categorical", "n_obs": int(s.notna().sum()),
            "n_missing": int(s.isna().sum()), "n_unique": int(s.nunique()),
        })
    return pd.DataFrame(rows).round(4)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def save_figure(fig: plt.Figure, path: Path) -> Path:
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure -> %s", path)
    return path


def plot_class_distribution(df: pd.DataFrame, path: Path, target: str = TARGET_COL) -> Path:
    """Bar chart of target class counts with percentages."""
    dist = class_distribution(df, target)
    fig, ax = plt.subplots(figsize=(5, 4))
    positions = list(range(len(dist)))
    bars = ax.bar(positions, dist["count"],
                  color=[_COLORS.get(int(i), "grey") for i in dist.index])
    ax.set_xticks(positions)
    ax.set_xticklabels([str(i) for i in dist.index])
    for bar, pct in zip(bars, dist["pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{pct:.1f}%",
                ha="center", va="bottom")
    ax.set_title("Class distribution (0 = repaid, 1 = default)")
    ax.set_xlabel(target)
    ax.set_ylabel("customers")
    return save_figure(fig, path)


def plot_missing_values(df: pd.DataFrame, path: Path) -> Path:
    """Horizontal bar chart of missing percentage per column (only columns with gaps)."""
    pct = (df.isna().mean() * 100).loc[lambda s: s > 0].sort_values()
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.3 * max(len(pct), 1) + 1)))
    if pct.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.axis("off")
    else:
        ax.barh(pct.index, pct.values, color="#F58518")
        ax.set_xlabel("% missing")
    ax.set_title("Missing value overview")
    return save_figure(fig, path)


def plot_numeric_distributions(df: pd.DataFrame, columns: Sequence[str], path: Path) -> Path:
    """Grid of histograms for feature distribution checks."""
    cols = [c for c in columns if c in df.columns]
    n_cols = 4
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.8 * n_rows))
    for ax, col in zip(np.ravel(axes), cols):
        ax.hist(df[col].dropna(), bins=40, color="#4C78A8", edgecolor="white")
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in np.ravel(axes)[len(cols):]:
        ax.axis("off")
    fig.suptitle("Numeric feature distributions", y=1.0)
    return save_figure(fig, path)


def plot_correlation_heatmap(corr: pd.DataFrame, path: Path) -> Path:
    """Heatmap of the correlation matrix."""
    size = max(8, 0.45 * len(corr))
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    image = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_title("Correlation overview (Pearson)")
    return save_figure(fig, path)


def plot_default_rate_by_category(
    df: pd.DataFrame, columns: Sequence[str], path: Path, target: str = TARGET_COL
) -> Path:
    """One bar chart per categorical feature showing default rate by level."""
    cols = [c for c in columns if c in df.columns]
    n_cols = 3
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.2 * n_rows))
    overall = df[target].mean()
    for ax, col in zip(np.ravel(axes), cols):
        table = default_rate_by_category(df, col, target).dropna(subset=["level"])
        ax.bar(table["level"].astype(str), table["default_rate"] * 100, color="#E45756")
        ax.axhline(overall * 100, color="black", linestyle="--", linewidth=1)
        ax.set_title(col, fontsize=10)
        ax.set_ylabel("default rate %", fontsize=8)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
    for ax in np.ravel(axes)[len(cols):]:
        ax.axis("off")
    fig.suptitle("Default rate by category (dashed = portfolio average)", y=1.0)
    return save_figure(fig, path)


def plot_target_vs_numeric(
    df: pd.DataFrame, columns: Sequence[str], path: Path, target: str = TARGET_COL
) -> Path:
    """Box plots of key risk variables split by target class."""
    cols = [c for c in columns if c in df.columns]
    n_cols = 3
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows))
    for ax, col in zip(np.ravel(axes), cols):
        groups = [df.loc[df[target] == k, col].dropna() for k in (0, 1)]
        ax.boxplot(groups, showfliers=False)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["repaid", "default"])
        ax.set_title(col, fontsize=10)
    for ax in np.ravel(axes)[len(cols):]:
        ax.axis("off")
    fig.suptitle("Key risk variables by outcome", y=1.0)
    return save_figure(fig, path)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_eda(
    df: pd.DataFrame,
    *,
    figures_dir: Path | str = PATHS.figures,
    summaries_dir: Path | str = PATHS.summaries,
    target: str = TARGET_COL,
) -> dict[str, Path]:
    """Generate the full EDA bundle (tables + plots) and return their paths.

    Args:
        df: Cleaned (optionally feature-engineered) dataset with a 0/1 target.
        figures_dir: Where PNG plots are written.
        summaries_dir: Where CSV / JSON summaries are written.
        target: Name of the target column.
    """
    figures_dir, summaries_dir = ensure_dir(figures_dir), ensure_dir(summaries_dir)
    numeric = [c for c in (*NUMERIC_COLUMNS, *ENGINEERED_NUMERIC_COLUMNS) if c in df.columns]
    categorical = [c for c in (*CATEGORICAL_COLUMNS, *ENGINEERED_CATEGORICAL_COLUMNS) if c in df.columns]
    outputs: dict[str, Path] = {}

    # Tables
    class_distribution(df, target).to_csv(summaries_dir / "class_distribution.csv")
    numeric_summary(df, numeric).to_csv(summaries_dir / "numeric_summary.csv")
    corr = correlation_matrix(df, [*numeric, target])
    corr.to_csv(summaries_dir / "correlation_matrix.csv")
    top_corr = target_correlations(df, numeric, target)
    top_corr.rename("corr_with_target").to_csv(summaries_dir / "target_correlations.csv")
    pd.concat([default_rate_by_category(df, c, target) for c in categorical]).to_csv(
        summaries_dir / "default_rate_by_category.csv", index=False
    )
    (df.isna().sum().rename("missing_count").to_frame()
       .assign(missing_pct=lambda t: (t["missing_count"] / len(df) * 100).round(3))
       .sort_values("missing_count", ascending=False)
       .to_csv(summaries_dir / "missing_values_cleaned.csv"))
    compute_feature_summary(df, numeric, categorical).to_csv(
        summaries_dir / "feature_summary.csv", index=False
    )
    outputs["summaries_dir"] = summaries_dir

    # Plots
    outputs["class_distribution"] = plot_class_distribution(df, figures_dir / "class_distribution.png", target)
    outputs["missing_values"] = plot_missing_values(df, figures_dir / "missing_values.png")
    outputs["numeric_distributions"] = plot_numeric_distributions(df, numeric, figures_dir / "numeric_distributions.png")
    outputs["correlation_heatmap"] = plot_correlation_heatmap(corr, figures_dir / "correlation_heatmap.png")
    outputs["default_rate_by_category"] = plot_default_rate_by_category(df, categorical, figures_dir / "default_rate_by_category.png", target)
    outputs["target_vs_numeric"] = plot_target_vs_numeric(df, KEY_RISK_VARIABLES, figures_dir / "target_vs_numeric.png", target)

    save_json(
        {
            "n_rows": len(df),
            "default_rate_pct": round(float(df[target].mean() * 100), 2),
            "top_target_correlations": top_corr.head(8).to_dict(),
        },
        summaries_dir / "eda_highlights.json",
    )
    logger.info("EDA complete: %d plots, tables in %s", len(outputs) - 1, summaries_dir)
    return outputs
