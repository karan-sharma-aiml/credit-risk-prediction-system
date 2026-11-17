"""Liveness / readiness."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from backend.app.core.config import APP_NAME, APP_VERSION

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health", summary="Health check (database + model)")
def health(request: Request) -> JSONResponse:
    """Return 200 when the API can reach the database and the model is loaded, otherwise 503."""
    state = request.app.state
    try:
        with state.session_factory() as session:
            session.execute(text("SELECT 1"))
        database_ok = True
    except Exception:
        logger.exception("health_database_check_failed")
        database_ok = False
    model_loaded = state.model_context is not None
    healthy = database_ok and model_loaded
    body = {
        "status": "ok" if healthy else "degraded",
        "app": APP_NAME,
        "version": APP_VERSION,
        "environment": state.settings.environment,
        "database": "up" if database_ok else "down",
        "model": state.model_context.predictor.model_version if model_loaded else "not_loaded",
    }
    return JSONResponse(body, status_code=200 if healthy else 503)
