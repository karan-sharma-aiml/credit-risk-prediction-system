# Final report: Credit Risk Prediction Platform (Part 3)

## 1. Completed features

| Area | Delivered |
|---|---|
| PostgreSQL integration | SQLAlchemy 2 ORM (`customers`, `predictions`, `prediction_history`, `model_registry`, `model_metrics`), env-driven config, pooled engine (`pool_pre_ping`, size/overflow/timeout/recycle), request-scoped sessions, repository (CRUD) layer, idempotent table creation |
| FastAPI backend | 12 endpoints, factory + lifespan, dependency injection, error envelope, request-id middleware, CORS, OpenAPI docs |
| Validation | Pydantic v2 schemas for age, income, loan amount, employment, credit score, savings, expenses, loan term, debt (+ cross-field rule); errors never echo input |
| Prediction service | validate -> `CreditRiskPredictor.predict_one` -> persist customer + prediction + audit row (one transaction, rollback on failure) -> respond |
| Persistence | Every prediction stores customer, probability, label, segment, threshold, recommendation, model version, timestamp, request id, and an input snapshot with change vs. the previous prediction |
| Recommendation engine | Policy (Approve / Manual Review / Reject) + replaceable consideration rules |
| Streamlit dashboard | 5 pages, 9 chart types (risk distribution, income / loan / credit-score distributions, prediction trend, feature importance, ROC, PR, confusion matrix) + gauge |
| Logging | JSON structured logs: requests, predictions, DB lifecycle, model loading, exceptions; customer ids hashed, routes templated, no bodies / SQL parameters |
| Configuration | `.env` / `.env.example` (`DATABASE_URL`, `MODEL_PATH`, `SECRET_KEY`, `LOG_LEVEL`, `API_HOST`, `API_PORT`, pool settings, ...); production refuses the default `SECRET_KEY` |
| Docker | `Dockerfile` (API), `Dockerfile.dashboard`, `docker-compose.yml` (PostgreSQL + API + dashboard + optional seed profile) |
| Deployment | `render.yaml`, `railway.toml`, `.github/workflows/ci.yml`, pinned `requirements*.txt` |
| Tooling | `backend/app/scripts/seed_demo.py` (demo data through the real service layer) |
| Docs | README, `docs/api.md`, `docs/deployment.md`, updated SQL README |

The ML layer was **not modified**: `model_bundle.joblib`, `preprocessor.joblib`, `classifier.joblib`, `model_metadata.json` and
`reports/summaries/test_predictions.csv` are byte-identical to the Part 2 delivery (checked with md5). Only `ml/config.py`'s
`SegmentSettings` thresholds (0.10 / 0.30) are used, unchanged.

## 2. Architecture

```
Streamlit (dashboard/) --HTTP--> FastAPI (backend/app) --SQLAlchemy--> PostgreSQL (schema credit_risk)
                                     |
                                     +--> ml.predict.CreditRiskPredictor + saved_models/model_bundle.joblib
```

`api/routers` -> `services` -> `database/crud` -> `database/models`; shared state on `app.state`, injected through `Depends`.
No scoring or preprocessing logic exists in the backend (it reuses `CreditRiskPredictor`, `clean_for_inference`,
`engineer_features`). `create_app(settings, predictor=..., engine=...)` allows full injection in tests.

## 3. API endpoints

`GET /health` · `GET /model-info` · `GET /model-info/performance` (extra) · `POST /predict` · `POST|GET /customers` ·
`GET|PUT|DELETE /customers/{id}` · `GET /predictions` · `GET /predictions/{id}` · `GET /dashboard`. Details: `docs/api.md`.

## 4. Database schema

| Table | Purpose |
|---|---|
| `customers` | Master data (model input schema), observed outcome `is_default`, source, timestamps; CHECK constraints on every range / category |
| `model_registry` | Model versions; one active per name (partial unique index) |
| `predictions` | Append-only outcomes (FK customer `ON DELETE CASCADE`, FK model) |
| `prediction_history` | Audit trail per prediction: `input_snapshot` JSON, previous prediction / probability / segment, `segment_changed`, `probability_change` |
| `model_metrics` | Validation / test metrics of registered models, long format (unique per model, split, metric) |
| `feature_summaries` | SQL-only (Part 1); loadable with `COPY` |

`sql/schema/*.sql` were executed on PostgreSQL 16 and compared with the ORM metadata (identical columns, enforced by a test).

## 5. Dashboard features

Dashboard (KPIs, risk donut, trend, recent predictions) · Customer Management (browse / search / create / edit / delete) ·
Prediction (new or existing customer, gauge, segment, recommendation, risk-migration note, field-level API errors) ·
Analytics (portfolio distributions; model metrics, feature importance, ROC, PR, confusion matrix) ·
Prediction History (filters, audit snapshot, CSV download).

## 6. Testing results

| Suite | Tests | Result |
|---|---|---|
| ML (Parts 1-2: data, cleaning, features, split, SMOTE, training, prediction, segmentation) | 55 | pass |
| Backend (API, CRUD, validation, recommendation, logging/privacy, config, schema parity, seed, PostgreSQL) | 77 | pass |
| Dashboard (AppTest on every page, API client, charts) | 19 | pass |
| **Total** | **151** | **151 passed** with `TEST_DATABASE_URL` set (148 passed + 3 skipped without PostgreSQL); pyflakes clean |

End-to-end (no Docker): production-mode `uvicorn` (exact Dockerfile `CMD`) on real PostgreSQL 16, 274 customers / 301
predictions seeded through the service layer, dashboard endpoint, live prediction, Streamlit headless server (`/_stcore/health`
ok) and **all five pages rendered with zero exceptions against the real API**. The API was also started from a directory
containing only the files the Dockerfile copies, and both requirement files resolve with their pins.

Defects found and fixed by this verification: (1) integer features (`np.int64`) were dropped when deriving recommendation
inputs, so the "late payments" rule never fired; (2) access logs contained raw URL paths with customer ids (now route
templates); (3) demo seed could create future-dated predictions; (4) the daily trend is now grouped by UTC day on PostgreSQL.

## 7. Known limitations

- **Docker, Docker Compose, Render, Railway and the GitHub Actions workflow were not executed**: the sandbox has no Docker
  daemon or CI runner. They were validated statically (YAML/TOML parse, COPY sources exist, image-equivalent file set boots).
  Run `docker compose up --build` once and the CI workflow once before relying on them.
- **No authentication / rate limiting**: any client that can reach the API can read and delete customers. Add auth before exposing it.
- The dashboard was verified with Streamlit's `AppTest` and a live headless server, but its visual layout was not inspected in a browser.
- Model quality is that of Part 2 on **synthetic data** (test ROC-AUC ~0.77); probabilities are approximately calibrated
  (the champion's mean predicted probability was ~9.6% vs 13.7% observed). The High Risk segment is ~2-3% of customers.
- `DELETE /customers/{id}` is a hard delete that cascades to predictions and audit rows (no soft delete / retention policy).
- Tables are created with `create_all`, not migrations (Alembic is listed as a next step); `PUT` is a full replacement and
  `POST /predict` with an existing `customer_id` overwrites the stored customer data (the previous inputs remain only in older
  audit snapshots).
- Dashboard histograms are computed in Python from the column values (fine to ~100k customers; use SQL width-bucket beyond that).
- Logging protections cover the application's loggers; running uvicorn *without* `--no-access-log` re-enables uvicorn's own
  access log, which prints raw URLs.
- Seed data is the synthetic Part 1 dataset; about 9% of its rows are skipped because they miss required fields.
