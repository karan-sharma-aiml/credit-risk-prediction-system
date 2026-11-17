# SQL schema

PostgreSQL 14+ DDL. The FastAPI backend (Part 3) maps these tables with SQLAlchemy ORM models
(`backend/app/database/models.py`) and creates them automatically on startup when
`AUTO_CREATE_TABLES=true`; a test fails if the ORM and these files drift apart. Apply the
files manually when you manage the schema yourself (e.g. `AUTO_CREATE_TABLES=false`).

| Order | File | Purpose |
|------|------|---------|
| 0 | `000_schema.sql` | `credit_risk` schema |
| 1 | `001_customers.sql` | Customer master data |
| 2 | `002_model_registry.sql` | Model versions (supports 3 and 4) |
| 3 | `003_predictions.sql` | Prediction outputs and risk segments |
| 4 | `004_model_metrics.sql` | Evaluation metrics per model / split |
| 5 | `005_feature_summaries.sql` | Feature statistics per dataset version |
| 6 | `006_prediction_history.sql` | Audit trail: exact inputs of each prediction + risk change vs. previous |

Apply in order:

```bash
for f in sql/schema/*.sql; do psql "$DATABASE_URL" -f "$f"; done
```

Notes:
- `default` is a reserved SQL keyword, so the target is stored as `is_default`.
- `feature_summaries` is SQL-only (no ORM model); load it with `COPY` from the CSV.
- `feature_summaries` columns match `reports/summaries/feature_summary.csv`.
