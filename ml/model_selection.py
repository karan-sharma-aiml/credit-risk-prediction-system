"""Champion-model selection rule.

Rule (documented, configurable in ``ml.config.SELECTION_SETTINGS``):

1. Every model gets a decision threshold chosen on the validation set that
   maximises F2 (recall-weighted) while keeping precision >= ``min_precision``.
2. Models are scored on **validation** data with
   ``0.40 * recall + 0.30 * F1 + 0.30 * ROC-AUC``.
3. The highest score wins; ties break on ROC-AUC, then PR-AUC.

Accuracy is deliberately excluded: with ~14% defaults, "approve everyone"
already scores ~86% accuracy. The test set plays no part in selection, so the
reported test metrics are an unbiased estimate for the chosen model.
"""

from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

from ml.config import SELECTION_SETTINGS, SelectionSettings
from ml.evaluate import ModelEvaluation

logger = logging.getLogger(__name__)


def selection_score(validation_metrics: dict, settings: SelectionSettings = SELECTION_SETTINGS) -> float:
    """Weighted business score from validation metrics."""
    return float(sum(weight * validation_metrics[name] for name, weight in settings.weights))


def rank_models(
    evaluations: Sequence[ModelEvaluation], settings: SelectionSettings = SELECTION_SETTINGS
) -> pd.DataFrame:
    """Rank models by selection score (best first).

    Returns:
        DataFrame with ``rank, model, selection_score`` and the metrics used.
    """
    if not evaluations:
        raise ValueError("No evaluations to rank.")
    rows = [
        {
            "model": ev.model_name,
            "selection_score": selection_score(ev.validation, settings),
            "val_recall": ev.validation["recall"],
            "val_f1": ev.validation["f1"],
            "val_roc_auc": ev.validation["roc_auc"],
            "val_pr_auc": ev.validation["pr_auc"],
        }
        for ev in evaluations
    ]
    ranking = pd.DataFrame(rows).sort_values(
        ["selection_score", "val_roc_auc", "val_pr_auc"], ascending=False
    ).reset_index(drop=True)
    ranking.insert(0, "rank", ranking.index + 1)
    return ranking.round(4)


def select_best_model(
    evaluations: Sequence[ModelEvaluation], settings: SelectionSettings = SELECTION_SETTINGS
) -> tuple[str, pd.DataFrame]:
    """Return ``(best_model_name, ranking_table)``."""
    ranking = rank_models(evaluations, settings)
    best = str(ranking.loc[0, "model"])
    logger.info("Selected model: %s (score %.4f)", best, ranking.loc[0, "selection_score"])
    return best, ranking
