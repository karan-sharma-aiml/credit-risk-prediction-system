# Credit Risk Prediction & Customer Segmentation System

Production-style credit-risk platform: ML pipeline (Parts 1-2) plus a **FastAPI + PostgreSQL backend, Streamlit
dashboard, Docker setup and deployment configs (Part 3)**. The model is trained once and served as-is.

## Business goal

Build the base for a system that will:

- predict customer credit risk (probability of default),
- classify customers into risk segments,
- support cleaning, preprocessing and feature engineering,
- produce structured outputs for model training and SQL storage.

## Part 1: data foundation

| Area | Location |
|---|---|
| Config / schema (single source of truth) | `ml/config.py` |
| Dataset generator, loader, validation | `ml/data/` |
| Stateless cleaning + sklearn preprocessing | `ml/preprocessing/` |
| Financial feature engineering | `ml/features.py` |
| EDA tables + plots | `ml/eda.py`, `reports/` |
| End-to-end pipeline + CLI | `ml/pipeline.py` |
| SQL schema (design only) | `sql/schema/` |
| Tests | `tests/` |
| Documentation | `docs/` |

## Quick start

```bash
pip install -r requirements.txt
python -m ml.pipeline          # generates data if absent, cleans, engineers features, writes reports
pytest                         # run the test suite
```

Train the models (Part 2):

```bash
python -m ml.train               # all models, SMOTE on, saves artifacts + reports
python -m ml.train --tune        # lightweight randomized hyper-parameter search
python -m ml.train --models random_forest lightgbm --seed 7
```

Score a customer:

```python
from ml.predict import CreditRiskPredictor
CreditRiskPredictor.from_path().predict_one({...raw customer fields...})
# -> default_probability, predicted_label, risk_segment (low/medium/high), risk_label
```

Data pipeline outputs:

- `data/interim/credit_risk_clean.csv` - cleaned data
- `data/processed/credit_risk_features.csv` - cleaned data + 12 engineered features
- `reports/figures/*.png` - class balance, missing values, distributions, correlations, default rate by segment, target vs. risk variables
- `reports/summaries/*.csv|json` - data-quality, cleaning, correlation, default-rate and feature-summary tables

## Part 2: ML layer

| Area | Location |
|---|---|
| Stratified train/validation/test split | `ml/splitting.py` |
| SMOTE (train only), prior-shift probability correction | `ml/imbalance.py` |
| Model registry (LR, tree, RF, GB, XGBoost/LightGBM if installed) + tuning grids | `ml/models.py` |
| Training entry point | `ml/train.py` |
| Metrics, threshold selection, ROC/PR/confusion plots | `ml/evaluate.py` |
| Champion selection rule | `ml/model_selection.py` |
| Risk segments (cut-offs in `ml/config.py`) | `ml/segmentation.py` |
| Model bundle save/load | `ml/artifacts.py` |
| Raw data -> probability, label, segment | `ml/predict.py` |

Design, selection rule, artifacts and a results snapshot: [`docs/modeling.md`](docs/modeling.md).

## Part 3: application layer

### Architecture

```
 Streamlit dashboard  --HTTP-->  FastAPI backend  --SQLAlchemy-->  PostgreSQL (schema credit_risk)
   (dashboard/)                   (backend/app/)                    customers, predictions,
                                       |                            prediction_history, model_registry,
                                       v                            model_metrics
                          ml.predict.CreditRiskPredictor
                          + saved_models/model_bundle.joblib   (Part 2 artifacts, never retrained)
```

Layers inside the backend: `api/routers` (HTTP) -> `services` (use-cases, recommendation engine) -> `database/crud`
(repositories) -> `database/models` (ORM). Dependencies are injected with FastAPI `Depends` (state lives on `app.state`,
no module-level singletons). Prediction always goes through `CreditRiskPredictor.predict_one`; the backend contains no
preprocessing or scoring logic of its own.

### Folder structure

```
backend/app/   main.py (factory + lifespan) | api/routers | schemas | services | database/{models,session,crud}
               dependencies | core/{config,logging,exceptions} | middleware | scripts/seed_demo.py
dashboard/     app.py | views/{dashboard,customers,prediction,analytics,history}.py | charts.py | forms.py | api_client.py
ml/ utils/ sql/ saved_models/ reports/ tests/ docs/      (Parts 1-2)
Dockerfile  Dockerfile.dashboard  docker-compose.yml  render.yaml  railway.toml  .github/workflows/ci.yml
```

### Installation and running locally

```bash
pip install -r requirements-dev.txt        # API + dashboard + test dependencies
cp .env.example .env                       # edit; SQLite works out of the box: DATABASE_URL=sqlite:///./credit_risk.db
uvicorn backend.app.main:app_factory --factory --reload     # API: http://localhost:8000/docs
streamlit run dashboard/app.py                              # dashboard: http://localhost:8501
python -m ml.pipeline && python -m backend.app.scripts.seed_demo --n 300   # optional demo data
```

Environment variables: see [`docs/deployment.md`](docs/deployment.md#environment-variables) and `.env.example`.

### Database setup

Tables are created automatically on startup (`AUTO_CREATE_TABLES=true`, idempotent), including the `credit_risk` schema on
PostgreSQL. To manage the schema yourself run `sql/schema/*.sql` in order (`sql/README.md`). A test asserts the ORM models and
the SQL files define identical columns, and the SQL files are also applied to a real PostgreSQL in the integration tests.
The model is registered in `model_registry` at startup and its validation/test metrics are stored in `model_metrics`.

### Streamlit usage

Pages: **Dashboard** (KPIs, risk distribution, prediction trend, recent predictions), **Customer Management**
(browse/search, create, edit, delete), **Prediction** (new or existing customer; gauge, segment, recommendation),
**Analytics** (income, loan and credit-score distributions; feature importance, ROC, precision-recall, confusion matrix),
**Prediction History** (filters, audit snapshot, CSV export). The dashboard talks only to the API (`API_URL`).

### Docker

```bash
cp .env.example .env     # set SECRET_KEY and POSTGRES_PASSWORD
docker compose up --build
```

PostgreSQL + API (`:8000`) + dashboard (`:8501`); details and the optional `seed` profile in [`docs/deployment.md`](docs/deployment.md).

### API examples

```bash
curl localhost:8000/health
curl localhost:8000/model-info
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d @customer.json
curl "localhost:8000/predictions?risk_segment=high&limit=10"
curl localhost:8000/dashboard
```

Full endpoint table, request/response examples and error contract: [`docs/api.md`](docs/api.md).

### Deployment

Render (`render.yaml`), Railway (`railway.toml`) and GitHub Actions CI (`.github/workflows/ci.yml`: lint, tests with a PostgreSQL
service, Docker image builds). Guide and production checklist: [`docs/deployment.md`](docs/deployment.md).

### Logging and privacy

JSON logs with a per-request `X-Request-ID`. Customer ids appear only as keyed hashes, routes are logged as templates
(`/customers/{customer_id}`), request bodies, query strings and SQL parameters are never logged, and validation errors do not
echo submitted values. A test scans the log output for customer data.

### Testing

```bash
pytest                                                        # ML + API + database + dashboard (SQLite)
TEST_DATABASE_URL=postgresql://user:pass@localhost/db pytest  # additionally runs the PostgreSQL integration tests
```

### Future improvements

Authentication/authorisation (API keys or OAuth2) and rate limiting; Alembic migrations; soft-delete and retention policy for
customer data; SHAP-based per-prediction explanations; probability calibration (Platt/isotonic); batch scoring and async
jobs; model retraining/monitoring (drift using `feature_summaries`); multi-model A/B serving via `model_registry`;
pagination cursors for very large histories.

## Dataset

The brief asks for fields (income, savings, expenses, credit score, late
payments, ...) that no single public dataset contains, so Part 1 uses a
**reproducible synthetic portfolio** with realistic structure and deliberately
injected data problems. Full details, data dictionary, target definition and
assumptions: [`docs/dataset.md`](docs/dataset.md).

> **Important:** the default label comes from a hand-written risk formula, so
> model scores in Part 2 will show the pipeline works, not real-world accuracy.
> The loader supports real datasets via `load_dataset(path, column_map=...)`.

## Design principles

- **No leakage by construction.** Stateless steps (dedupe, range rules, feature formulas, fixed buckets) run before the split; stateful steps (clip bounds, imputation, scaling, encoding) live in an sklearn `ColumnTransformer` meant to be fit on training data only.
- **Missing stays missing** until the training-time imputer handles it; invalid values become `NaN`, not silent zeros.
- **One config.** Column names, ranges and thresholds live in `ml/config.py`; nothing is hardcoded in notebooks.
- **Quality bar:** PEP8, type hints, docstrings, logging, unit tests.

See [`docs/architecture.md`](docs/architecture.md) and
[`docs/feature_engineering.md`](docs/feature_engineering.md).

## Roadmap

- Part 1 done: data foundation. Part 2 done: models, evaluation, artifacts, inference.
- Part 3 done: FastAPI, PostgreSQL, Streamlit, Docker, deployment configs, integration tests (see `FINAL_REPORT.md`).
#   c r e d i t - r i s k - p r e d i c t i o n - s y s t e m  
 