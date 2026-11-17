"""Application factory and lifespan (startup / shutdown)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.api.router import api_router
from backend.app.core.config import APP_NAME, APP_VERSION, Settings, get_settings
from backend.app.core.exceptions import register_exception_handlers
from backend.app.core.logging import configure_logging
from backend.app.database.session import create_db_engine, create_session_factory, init_db, session_scope
from backend.app.middleware.request_logging import RequestLoggingMiddleware
from backend.app.services.model_service import ModelContext, ModelService, register_active_model
from backend.app.services.recommendation import RecommendationEngine
from ml.predict import CreditRiskPredictor

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    predictor: CreditRiskPredictor | None = None,
    engine: Engine | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    Args:
        settings: Configuration (defaults to environment / ``.env``).
        predictor: Pre-loaded predictor (tests); by default the bundle at
            ``MODEL_PATH`` is loaded with ``CreditRiskPredictor.from_path``.
        engine: Pre-built SQLAlchemy engine (tests); defaults to ``DATABASE_URL``.
    """
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level, settings.log_json)
        db_engine = engine or create_db_engine(settings)
        if settings.auto_create_tables:
            init_db(db_engine)
        factory: sessionmaker[Session] = create_session_factory(db_engine)

        app.state.settings = settings
        app.state.session_factory = factory
        app.state.recommender = RecommendationEngine()
        app.state.model_context = None
        app.state.model_service = None

        try:
            loaded = predictor or CreditRiskPredictor.from_path(settings.model_path)
            with session_scope(factory) as session:
                model_id = register_active_model(session, loaded, settings.model_path)
            app.state.model_context = ModelContext(loaded, model_id)
            app.state.model_service = ModelService(loaded, settings.reports_dir)
            logger.info("model_loaded", extra={"model_version": loaded.model_version})
        except Exception:
            logger.exception("model_load_failed")  # API stays up; /health reports 503

        yield

        db_engine.dispose()
        logger.info("shutdown_complete")

    app = FastAPI(
        title=APP_NAME,
        version=APP_VERSION,
        description="Predicts customer default risk, stores customers and prediction history.",
        lifespan=lifespan,
    )
    if settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware, allow_origins=settings.cors_origin_list, allow_methods=["*"], allow_headers=["*"]
        )
    app.add_middleware(RequestLoggingMiddleware)
    register_exception_handlers(app)
    app.include_router(api_router)
    return app


def app_factory() -> FastAPI:
    """ASGI factory: ``uvicorn backend.app.main:app_factory --factory``."""
    return create_app()
