import json

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE

from ml.config import BALANCING_SETTINGS, Paths, SelectionSettings
from ml.evaluate import ModelEvaluation, compute_metrics, select_threshold
from ml.imbalance import (
    PriorCorrection,
    balancing_report,
    build_training_pipeline,
    expected_resampled_counts,
)
from ml.model_selection import rank_models, select_best_model
from ml.models import get_model_registry
from ml.pipeline import get_feature_lists
from ml.preprocessing import build_preprocessor
from ml.splitting import split_data
from ml.train import run_training_pipeline, select_models


# --------------------------------------------------------------------------- split
def test_split_is_stratified_disjoint_and_sized(trained_run):
    s = trained_run.splits
    assert set(s.X_train.index).isdisjoint(s.X_val.index)
    assert set(s.X_train.index).isdisjoint(s.X_test.index)
    assert set(s.X_val.index).isdisjoint(s.X_test.index)
    total = len(s.y_train) + len(s.y_val) + len(s.y_test)
    assert abs(len(s.y_test) / total - 0.20) < 0.01
    assert abs(len(s.y_val) / total - 0.20) < 0.01
    rates = [s.y_train.mean(), s.y_val.mean(), s.y_test.mean()]
    assert max(rates) - min(rates) < 0.01


def test_split_rejects_bad_inputs():
    X, y = pd.DataFrame({"a": range(10)}), pd.Series([0] * 9 + [1])
    with pytest.raises(ValueError):
        split_data(X, y)


# ----------------------------------------------------------------------- balancing
def test_balancing_only_changes_training_distribution(trained_run):
    s = trained_run.splits
    report = balancing_report(s.y_train, s.y_val, s.y_test).set_index("split")
    train = report.loc["train"]
    assert train["default_after"] > train["default_before"]
    assert train["repaid_after"] == train["repaid_before"]
    for split in ("validation", "test"):
        assert report.loc[split, "default_after"] == report.loc[split, "default_before"]
        assert report.loc[split, "repaid_after"] == report.loc[split, "repaid_before"]


def test_expected_counts_match_actual_smote(trained_run):
    s = trained_run.splits
    numeric, categorical = get_feature_lists()
    X_t = build_preprocessor(numeric, categorical).fit_transform(s.X_train)
    _, y_res = SMOTE(sampling_strategy=BALANCING_SETTINGS.sampling_strategy, random_state=0).fit_resample(
        X_t, s.y_train
    )
    expected = expected_resampled_counts(s.y_train, BALANCING_SETTINGS.sampling_strategy)
    assert int(y_res.sum()) == expected["n_default"]
    assert int((y_res == 0).sum()) == expected["n_repaid"]


def test_smote_skipped_at_inference_and_validation_untouched(trained_run):
    s = trained_run.splits
    pipeline = trained_run.trained_models["logistic_regression"].pipeline
    assert "smote" in pipeline.named_steps
    assert pipeline.predict_proba(s.X_val).shape[0] == len(s.X_val)


def test_no_balance_pipeline_has_no_sampler():
    numeric, categorical = get_feature_lists()
    pipe = build_training_pipeline(
        build_preprocessor(numeric, categorical), get_model_registry()["logistic_regression"], balance=False
    )
    assert list(pipe.named_steps) == ["preprocessor", "model"]


def test_prior_correction_shrinks_inflated_probabilities():
    correction = PriorCorrection(train_prior=1 / 3, true_prior=0.14)
    p = np.array([0.1, 0.5, 0.9])
    corrected = correction.apply(p)
    assert (corrected < p).all() and (np.diff(corrected) > 0).all()
    identity = PriorCorrection.identity(0.2).apply(p)
    np.testing.assert_allclose(identity, p, atol=1e-6)


# ------------------------------------------------------------------------ training
def test_all_models_train_and_predict(trained_run):
    expected = set(get_model_registry())
    assert {"logistic_regression", "decision_tree", "random_forest", "gradient_boosting"} <= expected
    assert set(trained_run.trained_models) == expected
    X_val = trained_run.splits.X_val
    for name, trained in trained_run.trained_models.items():
        proba = trained.pipeline.predict_proba(X_val)[:, 1]
        assert proba.shape == (len(X_val),), name
        assert ((proba >= 0) & (proba <= 1)).all(), name


def test_select_models_validates_names():
    assert list(select_models(["decision_tree"], 1)) == ["decision_tree"]
    with pytest.raises(ValueError, match="Unknown models"):
        select_models(["not_a_model"], 1)


def test_tuning_runs_and_records_params(tmp_path):
    result = run_training_pipeline(
        paths=Paths.from_root(tmp_path), model_names=["decision_tree"], tune=True,
        n_rows=800, seed=3, run_plots=False,
    )
    trained = result.trained_models["decision_tree"]
    assert trained.best_params and trained.cv_score is not None


def test_training_is_reproducible(tmp_path):
    kwargs = dict(model_names=["logistic_regression", "decision_tree"], n_rows=800, seed=9, run_plots=False)
    a = run_training_pipeline(paths=Paths.from_root(tmp_path / "a"), **kwargs)
    b = run_training_pipeline(paths=Paths.from_root(tmp_path / "b"), **kwargs)
    for ea, eb in zip(a.evaluations, b.evaluations):
        assert ea.test == eb.test and ea.threshold == eb.threshold


# ---------------------------------------------------------------------- evaluation
def test_metrics_have_expected_format(trained_run):
    for evaluation in trained_run.evaluations:
        for split in (evaluation.validation, evaluation.test):
            for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score"):
                assert 0.0 <= split[key] <= 1.0
            cm = split["confusion_matrix"]
            assert set(cm) == {"tn", "fp", "fn", "tp"}
            assert sum(cm.values()) == (
                len(trained_run.splits.y_val) if split is evaluation.validation else len(trained_run.splits.y_test)
            )
        json.dumps(evaluation.to_dict())  # serialisable


def test_compute_metrics_on_known_values():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.9])
    m = compute_metrics(y, p, threshold=0.5)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["accuracy"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5)
    assert m["roc_auc"] == pytest.approx(0.75)


def test_threshold_respects_min_precision_and_favours_recall():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.15).astype(int)
    p = np.clip(0.15 + 0.35 * y + rng.normal(0, 0.15, 2000), 0.001, 0.999)
    strict = select_threshold(y, p, SelectionSettings(threshold_beta=2.0, min_precision=0.5))
    loose = select_threshold(y, p, SelectionSettings(threshold_beta=2.0, min_precision=0.0))
    assert compute_metrics(y, p, strict)["precision"] >= 0.5
    assert compute_metrics(y, p, loose)["recall"] >= compute_metrics(y, p, strict)["recall"]


# ------------------------------------------------------------------------ selection
def _fake_eval(name: str, recall: float, f1: float, auc: float) -> ModelEvaluation:
    metrics = {"recall": recall, "f1": f1, "roc_auc": auc, "pr_auc": 0.3}
    return ModelEvaluation(name, 0.3, metrics, metrics)


def test_best_model_prefers_recall_f1_auc_not_accuracy():
    evaluations = [
        _fake_eval("high_recall", recall=0.80, f1=0.40, auc=0.80),
        _fake_eval("high_precision", recall=0.40, f1=0.45, auc=0.82),
        _fake_eval("weak", recall=0.30, f1=0.30, auc=0.70),
    ]
    best, ranking = select_best_model(evaluations)
    assert best == "high_recall"
    assert ranking["model"].tolist() == ["high_recall", "high_precision", "weak"]
    assert ranking["rank"].tolist() == [1, 2, 3]


def test_ranking_breaks_ties_on_roc_auc_and_rejects_empty():
    tie = [_fake_eval("a", 0.5, 0.4, 0.70), _fake_eval("b", 0.5, 0.4, 0.80)]
    assert rank_models(tie).loc[0, "model"] == "b"
    with pytest.raises(ValueError):
        rank_models([])


def test_trained_run_selected_model_is_top_ranked(trained_run):
    assert trained_run.ranking.loc[0, "model"] == trained_run.best_model_name


# ------------------------------------------------------------------------ artifacts
def test_artifacts_and_reports_exist(trained_run):
    paths: Paths = trained_run.paths
    for key in ("bundle", "preprocessor", "classifier", "metadata"):
        assert trained_run.artifact_paths[key].is_file(), key
    for name in (
        "model_comparison.csv", "model_metrics.json", "model_metrics_long.csv", "model_ranking.csv",
        "test_predictions.csv", "segment_summary.csv", "split_summary.csv", "balancing_report.csv",
        "sample_inference.csv", "sample_inference.json",
    ):
        assert (paths.summaries / name).is_file(), name
    for name in (
        "model_roc_curves.png", "model_precision_recall_curves.png", "model_confusion_matrices.png",
        "model_probability_distribution.png", "model_comparison.png",
    ):
        assert (paths.figures / name).is_file(), name

    meta = json.loads(paths.model_metadata.read_text())
    assert meta["model_name"] == trained_run.best_model_name
    assert meta["random_state"] == 11


def test_metrics_long_table_matches_sql_contract(trained_run):
    long = pd.read_csv(trained_run.paths.summaries / "model_metrics_long.csv")
    assert list(long.columns) == ["model_name", "dataset_split", "metric_name", "metric_value"]
    assert set(long["dataset_split"]) == {"validation", "test"}
    assert {"recall", "roc_auc", "tp"} <= set(long["metric_name"])
