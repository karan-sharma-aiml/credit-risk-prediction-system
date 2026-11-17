"""Model training entry point (Part 2).

Run from the project root::

    python -m ml.train                       # all models, SMOTE on, defaults
    python -m ml.train --tune                # randomized hyper-parameter search
    python -m ml.train --models random_forest lightgbm --seed 7

Flow: features -> stratified split -> [preprocess -> SMOTE -> model] per model
-> evaluate (threshold on validation, report on test) -> select champion ->
save artifacts and reports.
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from ml import evaluate as ev
from ml.artifacts import ModelBundle, build_metadata, make_model_version, save_bundle
from ml.config import (
    BALANCING_SETTINGS,
    PATHS,
    RANDOM_STATE,
    SEGMENT_SETTINGS,
    SELECTION_SETTINGS,
    TARGET_COL,
    TUNING_SETTINGS,
    BalancingSettings,
    Paths,
    TuningSettings,
)
from ml.imbalance import PriorCorrection, balancing_report, build_training_pipeline
from ml.model_selection import select_best_model
from ml.models import PARAM_DISTRIBUTIONS, get_model_registry
from ml.pipeline import get_feature_lists, run_data_pipeline
from ml.predict import CreditRiskPredictor
from ml.preprocessing import build_preprocessor
from ml.segmentation import assign_risk_segment, summarize_segments
from ml.splitting import DataSplits, split_data
from utils.io_utils import save_json
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class TrainedModel:
    """A fitted pipeline plus how it was obtained."""

    name: str
    pipeline: ImbPipeline
    best_params: dict = field(default_factory=dict)
    cv_score: float | None = None
    fit_seconds: float = 0.0


@dataclass
class TrainingResult:
    """Everything produced by :func:`run_training_pipeline`."""

    splits: DataSplits
    trained_models: dict[str, TrainedModel]
    evaluations: list[ev.ModelEvaluation]
    ranking: pd.DataFrame
    best_model_name: str
    bundle: ModelBundle
    prior_correction: PriorCorrection
    artifact_paths: dict[str, Path]
    paths: Paths = PATHS
    report_paths: dict[str, Path] = field(default_factory=dict)


def set_global_seed(seed: int) -> None:
    """Seed Python and NumPy RNGs (estimators also receive ``random_state``)."""
    random.seed(seed)
    np.random.seed(seed)


def select_models(names: list[str] | None, seed: int) -> dict[str, BaseEstimator]:
    """Return the requested subset of the model registry.

    Raises:
        ValueError: If a requested name is not available.
    """
    registry = get_model_registry(seed)
    if not names:
        return registry
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise ValueError(f"Unknown models {unknown}. Available: {sorted(registry)}")
    return {n: registry[n] for n in names}


def train_model(
    name: str,
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
    balance: bool = True,
    tune: bool = False,
    seed: int = RANDOM_STATE,
    balancing: BalancingSettings = BALANCING_SETTINGS,
    tuning: TuningSettings = TUNING_SETTINGS,
) -> TrainedModel:
    """Fit ``preprocessor -> [SMOTE] -> estimator`` on training data only.

    With ``tune=True`` a randomized search runs on the *training* split; SMOTE
    and the preprocessor are refit inside every CV fold, so there is no
    leakage between folds.
    """
    pipeline = build_training_pipeline(
        build_preprocessor(numeric_columns, categorical_columns),
        clone(estimator),
        balance=balance,
        settings=balancing,
        seed=seed,
    )
    start = time.perf_counter()
    best_params: dict = {}
    cv_score: float | None = None

    if tune and name in PARAM_DISTRIBUTIONS:
        space = {f"model__{k}": v for k, v in PARAM_DISTRIBUTIONS[name].items()}
        grid_size = int(np.prod([len(v) for v in space.values()]))
        search = RandomizedSearchCV(
            pipeline,
            space,
            n_iter=min(tuning.n_iter, grid_size),
            cv=StratifiedKFold(tuning.cv_folds, shuffle=True, random_state=seed),
            scoring=tuning.scoring,
            random_state=seed,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        best_params = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
        cv_score = float(search.best_score_)
    else:
        pipeline.fit(X_train, y_train)

    seconds = time.perf_counter() - start
    logger.info("Trained %-20s in %.1fs%s", name, seconds, f" (tuned, cv={cv_score:.3f})" if cv_score else "")
    return TrainedModel(name, pipeline, best_params, cv_score, seconds)


def evaluate_trained_model(
    trained: TrainedModel, splits: DataSplits, correction: PriorCorrection
) -> ev.ModelEvaluation:
    """Predict corrected probabilities on validation/test and compute metrics."""
    proba_val = correction.apply(trained.pipeline.predict_proba(splits.X_val)[:, 1])
    proba_test = correction.apply(trained.pipeline.predict_proba(splits.X_test)[:, 1])
    return ev.evaluate_probabilities(
        trained.name, splits.y_val, proba_val, splits.y_test, proba_test, SELECTION_SETTINGS
    )


def _write_reports(
    paths: Paths,
    splits: DataSplits,
    ids_test: pd.Series,
    evaluations: list[ev.ModelEvaluation],
    ranking: pd.DataFrame,
    best_eval: ev.ModelEvaluation,
    balance: bool,
    trained: dict[str, TrainedModel],
    plots: bool,
) -> dict[str, Path]:
    """Write all tables and (optionally) plots into ``reports/``."""
    s, f = paths.summaries, paths.figures
    out: dict[str, Path] = {}

    splits.class_distribution().to_csv(s / "split_summary.csv", index=False)
    balancing_report(splits.y_train, splits.y_val, splits.y_test, balance=balance).to_csv(
        s / "balancing_report.csv", index=False
    )
    table = ev.comparison_table(evaluations)
    table = table.merge(ranking[["model", "rank", "selection_score"]], on="model").sort_values("rank")
    table["selected"] = table["model"] == best_eval.model_name
    table.to_csv(s / "model_comparison.csv", index=False)
    ranking.to_csv(s / "model_ranking.csv", index=False)
    ev.metrics_long_table(evaluations).to_csv(s / "model_metrics_long.csv", index=False)

    save_json(
        {
            "best_model": best_eval.model_name,
            "selection_rule": (
                "threshold = argmax F2 on validation with precision >= min_precision; "
                "score = 0.40*recall + 0.30*F1 + 0.30*ROC-AUC on validation"
            ),
            "models": {
                e.model_name: {
                    **e.to_dict(),
                    "best_params": trained[e.model_name].best_params,
                    "cv_score": trained[e.model_name].cv_score,
                    "fit_seconds": round(trained[e.model_name].fit_seconds, 2),
                }
                for e in evaluations
            },
        },
        s / "model_metrics.json",
    )

    proba = best_eval.probabilities["test"]
    predictions = pd.DataFrame({
        "customer_id": ids_test.to_numpy(),
        "actual_default": splits.y_test.to_numpy(),
        "default_probability": np.round(proba, 6),
        "predicted_label": (proba >= best_eval.threshold).astype(int),
    })
    predictions = pd.concat([predictions, assign_risk_segment(proba, SEGMENT_SETTINGS, best_eval.threshold).reset_index(drop=True)], axis=1)
    predictions.to_csv(s / "test_predictions.csv", index=False)
    summarize_segments(proba, splits.y_test, SEGMENT_SETTINGS, best_eval.threshold).to_csv(s / "segment_summary.csv", index=False)
    out.update(model_comparison=s / "model_comparison.csv", test_predictions=s / "test_predictions.csv")

    if plots:
        out["roc"] = ev.plot_roc_curves(evaluations, splits.y_test, f / "model_roc_curves.png", best_eval.model_name)
        out["pr"] = ev.plot_precision_recall_curves(
            evaluations, splits.y_test, f / "model_precision_recall_curves.png", best_eval.model_name
        )
        out["confusion"] = ev.plot_confusion_matrices(evaluations, f / "model_confusion_matrices.png")
        out["probability_distribution"] = ev.plot_probability_distribution(
            proba, splits.y_test, best_eval.threshold, f / "model_probability_distribution.png",
            f"Predicted default probability - {best_eval.model_name} (test set)",
        )
        out["comparison"] = ev.plot_model_comparison(table, f / "model_comparison.png")
    return out


def _write_sample_inference(paths: Paths, predictor: CreditRiskPredictor, n: int = 20) -> dict[str, Path]:
    """Score raw rows through the saved model to demonstrate end-to-end inference."""
    raw = pd.read_csv(paths.raw_dataset).head(n)
    result = predictor.predict(raw)
    csv_path = paths.summaries / "sample_inference.csv"
    result.to_csv(csv_path, index=False)

    record = raw.iloc[0].astype(object).where(raw.iloc[0].notna(), None).to_dict()
    json_path = save_json(
        {"request": record, "response": predictor.predict_one(record)},
        paths.summaries / "sample_inference.json",
    )
    return {"sample_inference_csv": csv_path, "sample_inference_json": json_path}


def run_training_pipeline(
    *,
    paths: Paths = PATHS,
    model_names: list[str] | None = None,
    balance: bool = True,
    tune: bool = False,
    seed: int = RANDOM_STATE,
    n_rows: int = 10_000,
    run_plots: bool = True,
) -> TrainingResult:
    """Train, evaluate, select and persist credit-risk models.

    Args:
        paths: Project directories (override in tests).
        model_names: Subset of the registry; ``None`` trains every model.
        balance: Apply SMOTE to the training split.
        tune: Run randomized hyper-parameter search per model.
        seed: Global random seed (recorded in the artifact metadata).
        n_rows: Synthetic rows to generate if no feature file exists yet.
        run_plots: Disable to skip figure generation (faster tests).
    """
    set_global_seed(seed)
    paths.ensure()

    if not paths.feature_dataset.is_file():
        logger.info("Feature dataset missing; running data pipeline first")
        run_data_pipeline(paths=paths, n_rows=n_rows, seed=seed, run_eda_step=False)
    data = pd.read_csv(paths.feature_dataset)

    numeric, categorical = get_feature_lists()
    ids = data["customer_id"]
    splits = split_data(data[numeric + categorical], data[TARGET_COL].astype(int), seed=seed)
    correction = PriorCorrection.from_counts(splits.y_train, balance=balance)
    logger.info("Prior correction: train_prior=%.3f -> true_prior=%.3f", correction.train_prior, correction.true_prior)

    trained = {
        name: train_model(name, est, splits.X_train, splits.y_train, numeric_columns=numeric,
                          categorical_columns=categorical, balance=balance, tune=tune, seed=seed)
        for name, est in select_models(model_names, seed).items()
    }
    evaluations = [evaluate_trained_model(t, splits, correction) for t in trained.values()]
    best_name, ranking = select_best_model(evaluations, SELECTION_SETTINGS)
    best_eval = next(e for e in evaluations if e.model_name == best_name)

    if best_eval.threshold >= SEGMENT_SETTINGS.high_min:
        logger.warning(
            "Decision threshold %.3f >= high-risk cut-off %.2f; review SEGMENT_SETTINGS",
            best_eval.threshold, SEGMENT_SETTINGS.high_min,
        )
    bundle = ModelBundle(
        model_name=best_name,
        model_version=make_model_version(best_name),
        pipeline=trained[best_name].pipeline,
        prior_correction=correction,
        decision_threshold=best_eval.threshold,
        segment_settings=SEGMENT_SETTINGS,
        numeric_features=numeric,
        categorical_features=categorical,
        metadata=build_metadata(
            {"balanced_with_smote": balance, "tuned": tune, "best_params": trained[best_name].best_params},
            seed=seed,
            train_rows=len(splits.y_train),
            metrics={"validation": best_eval.validation, "test": best_eval.test},
        ),
    )
    artifact_paths = save_bundle(bundle, paths)

    report_paths = _write_reports(
        paths, splits, ids.loc[splits.X_test.index], evaluations, ranking, best_eval, balance, trained, run_plots
    )
    if paths.raw_dataset.is_file():
        report_paths.update(_write_sample_inference(paths, CreditRiskPredictor(bundle)))

    logger.info(
        "Done. Best=%s | test recall=%.3f precision=%.3f F1=%.3f ROC-AUC=%.3f",
        best_name, best_eval.test["recall"], best_eval.test["precision"],
        best_eval.test["f1"], best_eval.test["roc_auc"],
    )
    return TrainingResult(splits, trained, evaluations, ranking, best_name, bundle,
                          correction, artifact_paths, paths, report_paths)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Train and evaluate credit-risk models.")
    parser.add_argument("--models", nargs="*", help="Subset of models (default: all available)")
    parser.add_argument("--tune", action="store_true", help="Randomized hyper-parameter search")
    parser.add_argument("--no-balance", action="store_true", help="Disable SMOTE")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--n-rows", type=int, default=10_000)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    run_training_pipeline(
        model_names=args.models, balance=not args.no_balance, tune=args.tune,
        seed=args.seed, n_rows=args.n_rows, run_plots=not args.skip_plots,
    )


if __name__ == "__main__":
    main()
