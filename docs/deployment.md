# Deployment guide

The shipped model (`saved_models/model_bundle.joblib`, ~9.6 MB) is part of the repository and is **served, never retrained**,
by every deployment target. Library versions in `requirements.txt` are pinned to the versions the model was trained with
(unpickling across scikit-learn/numpy versions is not guaranteed). Use Git LFS if artifacts grow.

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./credit_risk.db` | Any SQLAlchemy URL; `postgres://` / `postgresql://` are normalised to `postgresql+psycopg2://` |
| `MODEL_PATH` | `saved_models/model_bundle.joblib` | Relative paths resolve from the project root |
| `SECRET_KEY` | `change-me` | HMAC key that pseudonymises customer ids in logs. **Startup fails in `production` if left at the default** |
| `ENVIRONMENT` | `development` | `development` / `test` / `production` |
| `LOG_LEVEL` | `INFO` | |
| `LOG_JSON` | `true` | `false` = human-readable logs |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | `PORT` (Render/Railway) takes precedence in the container command |
| `CORS_ORIGINS` | empty | Comma separated; only needed for browser clients calling the API directly |
| `AUTO_CREATE_TABLES` | `true` | Create the `credit_risk` schema/tables on startup (idempotent) |
| `DB_POOL_SIZE`, `DB_MAX_OVERFLOW`, `DB_POOL_TIMEOUT`, `DB_POOL_RECYCLE` | 5, 10, 30, 1800 | PostgreSQL connection pool |
| `API_URL` | `http://localhost:8000` | Dashboard only. A bare `host:port` is accepted (Render private network) |
| `POSTGRES_USER/PASSWORD/DB` | none | Docker Compose only; no defaults on purpose |

Special characters in a password must be URL-encoded inside `DATABASE_URL`.

## Docker Compose

```bash
cp .env.example .env          # set SECRET_KEY, POSTGRES_PASSWORD (compose refuses to start without them)
docker compose up --build     # db -> api (:8000) -> dashboard (:8501)
python -m ml.pipeline         # only if you want demo data (creates data/interim/...)
docker compose --profile seed run --rm seed
```

API container: non-root user, `/health` HEALTHCHECK, `uvicorn ... --no-access-log` (the app does its own access logging
with route templates, because raw URLs contain customer ids). The dashboard container has no ML dependencies.

## Render

`render.yaml` is a Blueprint (API + dashboard + Postgres). In Render choose **New > Blueprint** and select the repository.
Build `pip install -r requirements.txt`; start `uvicorn backend.app.main:app_factory --factory --host 0.0.0.0 --port $PORT --no-access-log`;
health check `/health`; `SECRET_KEY` is generated; `DATABASE_URL` comes from the managed database; the dashboard receives the
API host through `fromService`. Free Postgres plans expire: use a paid plan for production.

## Railway

`railway.toml` builds the root `Dockerfile` for the API (health check `/health`). Add a PostgreSQL plugin and reference its
`DATABASE_URL`; set `SECRET_KEY` and `ENVIRONMENT=production`. For the dashboard create a second service from the same repo with
`RAILWAY_DOCKERFILE_PATH=Dockerfile.dashboard` and `API_URL` pointing to the API service.

## Production checklist

- Put the API behind HTTPS and **add authentication** (the API is currently open; see limitations).
- Use a managed Postgres with backups; consider `AUTO_CREATE_TABLES=false` and apply `sql/schema/*.sql` / migrations yourself.
- Monitor `/health`; ship the JSON logs to your log platform (`request_id` correlates API logs with `predictions.request_id`).
