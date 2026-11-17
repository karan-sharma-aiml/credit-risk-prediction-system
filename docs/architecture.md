# Architecture (Part 1)

```
credit-risk-system/
|-- ml/
|   |-- config.py            schema, paths, ranges, feature settings (single source of truth)
|   |-- data/                generator.py, loader.py, validation.py
|   |-- preprocessing/       cleaning.py (stateless), transformers.py (stateful, sklearn)
|   |-- features.py          financial feature engineering (+ sklearn wrapper)
|   |-- eda.py               summary tables + plots
|   |-- pipeline.py          data-stage orchestration + CLI (python -m ml.pipeline)
|   |-- splitting.py, imbalance.py, models.py      Part 2: split, SMOTE, model registry
|   |-- train.py, evaluate.py, model_selection.py  Part 2: training, metrics, champion rule
|   `-- segmentation.py, artifacts.py, predict.py  Part 2: segments, persistence, inference
|-- utils/                   logging_config.py, io_utils.py
|-- sql/schema/              PostgreSQL DDL (design only)
|-- tests/                   pytest suite
|-- data/{raw,interim,processed}/   pipeline outputs (git-ignored)
|-- reports/{figures,summaries}/    EDA outputs
|-- saved_models/            model_bundle.joblib, preprocessor/classifier extracts, metadata
`-- notebooks/, docs/
```

## Data flow

```
raw CSV --load/validate--> quality report
        --clean (stateless)--> data/interim/credit_risk_clean.csv
        --engineer features--> data/processed/credit_risk_features.csv
        --EDA--> reports/figures/*.png, reports/summaries/*.csv|json
```

## Leakage-safe split of responsibilities

| Step | Learns from data? | Where it runs |
|---|---|---|
| De-duplication, target cleanup, range rules, feature formulas, fixed buckets | No | Whole dataset, before the split |
| Outlier clip bounds, imputation values, scaling, one-hot levels | **Yes** | `build_preprocessor(...)`, fit on **train only** (Part 2) |

## Part 2 handoff (done - see docs/modeling.md)

```python
from ml.pipeline import get_feature_lists
from ml.preprocessing import build_preprocessor
from ml.config import TARGET_COL
import pandas as pd

df = pd.read_csv("data/processed/credit_risk_features.csv")
numeric, categorical = get_feature_lists()
X, y = df[numeric + categorical], df[TARGET_COL]
# 1) stratified train/test split
# 2) pre = build_preprocessor(numeric, categorical)   # fit on train only
# 3) class balancing, model comparison, evaluation, joblib -> saved_models/
# 4) segmentation thresholds -> predictions table (sql/schema/003)
```

Part 3 extension points: `api/` (FastAPI wraps `CreditRiskPredictor`), `dashboard/`
(Streamlit), `deploy/` (Docker) can be added as sibling packages without
touching existing modules, because every consumer imports schema constants from
`ml/config.py` and never hardcodes column names.
