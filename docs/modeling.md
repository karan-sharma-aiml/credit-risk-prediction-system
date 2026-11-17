# Modeling (Part 2)

## Pipeline

```
features CSV -> stratified split (60/20/20)
             -> per model: preprocessor -> SMOTE -> classifier   (fit on TRAIN only)
             -> probabilities -> prior correction
             -> threshold chosen on VALIDATION -> metrics on validation + TEST
             -> champion by validation score -> artifacts + reports
```

Entry point: `python -m ml.train` (`--tune`, `--models ...`, `--no-balance`, `--seed`).

## Split and leakage control

| Split | Used for |
|---|---|
| Train (60%) | Fitting preprocessor, SMOTE, models (and CV tuning) |
| Validation (20%) | Choosing each model's decision threshold and the champion |
| Test (20%) | Final, unbiased reporting only; never used for any decision |

Splits are stratified on the target. The preprocessor (clip bounds, medians, scaling,
one-hot levels) and SMOTE live inside one `imblearn.Pipeline`, so they are fit on
training rows only - also inside every CV fold when tuning.

## Why class balancing

Defaults are rare (~14% here; often 1-5% in production). Minimising overall error lets a
model reach ~86% accuracy by approving everyone, which is the costly failure in lending:
a missed defaulter loses principal, a wrongly declined good customer loses margin. SMOTE
(training split only, minority:majority raised to 1:2) gives the model enough defaulter
signal. Before/after counts are saved in `reports/summaries/balancing_report.csv`;
validation and test rows are never resampled.

**Probability correction.** Resampling inflates predicted probabilities (the model sees 33%
defaults instead of 14%). `PriorCorrection` (`ml/imbalance.py`) rescales the odds back to the true
prior, so outputs are readable as default probabilities and risk-segment cut-offs mean
something. It is monotonic, so ROC-AUC/PR-AUC are unchanged. It is an approximation: on this run the
champion's mean predicted probability was ~9.6% vs 13.7% observed (slightly low), so treat
probabilities as well-ordered scores with approximate calibration. Isotonic/Platt calibration is the
next step if exact probabilities matter.

## Selection rule (`ml/model_selection.py`, settings in `ml/config.py`)

1. **Threshold per model** - maximise F2 on validation (recall-weighted: a missed defaulter costs more
   than a false alarm) subject to precision >= 0.25, so "flag everyone" never wins.
2. **Score** = 0.40 x recall + 0.30 x F1 + 0.30 x ROC-AUC (validation).
3. Highest score wins; ties break on ROC-AUC, then PR-AUC. Accuracy is deliberately excluded.

## Risk segments (`ml/segmentation.py`)

Cut-offs live in `SEGMENT_SETTINGS`: **Low** `p < 0.10`, **Medium** `0.10 <= p < 0.30`, **High** `p >= 0.30`.
A customer the model flags (`p >= decision_threshold`) is never shown as Low, so
`predicted_label = 1` and "Low Risk" cannot contradict each other. Segment codes
(`low/medium/high`) match the SQL `predictions.risk_segment` constraint; `risk_label` is the display text.
`reports/summaries/segment_summary.csv` shows observed default rate per segment (should rise low -> high).

## Artifacts

| File | Content |
|---|---|
| `saved_models/model_bundle.joblib` | **Use this for inference**: pipeline + prior correction + threshold + segment settings + feature lists |
| `saved_models/preprocessor.joblib`, `classifier.joblib` | Fitted preprocessor / classifier extracted from the bundle |
| `saved_models/model_metadata.json` | Version, seed, thresholds, metrics, library versions (feeds SQL `model_registry`) |
| `reports/summaries/model_comparison.csv`, `model_ranking.csv`, `model_metrics.json` | Comparison and full metrics incl. confusion matrices |
| `reports/summaries/model_metrics_long.csv` | Long format, columns match SQL `model_metrics` |
| `reports/summaries/test_predictions.csv`, `segment_summary.csv` | Test-set scores with segments, segment quality |
| `reports/summaries/sample_inference.csv/.json` | Raw rows scored end-to-end; JSON is a ready API request/response example |
| `reports/figures/model_*.png` | ROC, precision-recall, confusion matrices, probability distribution, comparison |

## Inference

```python
from ml.predict import CreditRiskPredictor

predictor = CreditRiskPredictor.from_path()            # saved_models/model_bundle.joblib
predictor.predict_one({"age": 34, "annual_income": 62000, ...})  # raw schema, NaNs allowed
predictor.predict(raw_dataframe)                       # one output row per input row
```

Inference uses `clean_for_inference` (row-preserving: same type coercion and range rules as training,
never drops rows), `engineer_features`, then the stored pipeline. Unseen categories and missing values
are handled; a missing required column raises `ValueError`.

## Results snapshot (synthetic data, seed 42)

| Model | Threshold | Val recall | Val ROC-AUC | Test recall | Test precision | Test ROC-AUC | Rank |
|---|---|---|---|---|---|---|---|
| random_forest | 0.087 | 0.767 | 0.764 | 0.770 | 0.253 | 0.765 | 1 |
| gradient_boosting | 0.055 | 0.706 | 0.755 | 0.745 | 0.271 | 0.767 | 2 |
| logistic_regression | 0.124 | 0.687 | 0.757 | 0.759 | 0.279 | 0.785 | 3 |
| lightgbm | 0.040 | 0.684 | 0.744 | 0.697 | 0.283 | 0.758 | 4 |
| xgboost | 0.071 | 0.626 | 0.751 | 0.657 | 0.318 | 0.772 | 5 |
| decision_tree | 0.152 | 0.542 | 0.697 | 0.562 | 0.262 | 0.709 | 6 |

Read with care: the data is synthetic with a logistic risk formula plus noise, the top five models are
within noise of each other (validation has ~275 defaulters), and the champion was chosen on validation, so
test numbers are the honest estimate. A simple logistic regression is fully competitive here, which is
expected for this generator and worth re-checking on real data. Several thresholds sit at the precision
floor, i.e. the rule effectively maximises recall at ~25% precision.
