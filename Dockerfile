# FastAPI backend + ML inference. Build context = repository root.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# libgomp1: OpenMP runtime required by LightGBM / XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code + the trained artifacts that are served as-is (never retrained here)
COPY ml ./ml
COPY utils ./utils
COPY backend ./backend
COPY sql ./sql
COPY saved_models ./saved_models
COPY reports/summaries ./reports/summaries

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD ["python", "-c", "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/health' % os.getenv('PORT', os.getenv('API_PORT', '8000')), timeout=4)"]

# $PORT is injected by Render/Railway; falls back to API_PORT. Our middleware does the access logging.
CMD ["sh", "-c", "exec uvicorn backend.app.main:app_factory --factory --host ${API_HOST} --port ${PORT:-${API_PORT}} --no-access-log"]
