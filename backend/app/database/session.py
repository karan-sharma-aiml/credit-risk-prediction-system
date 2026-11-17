"""Engine, connection pooling and session management."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.core.config import Settings
from backend.app.database import models  # importing the module registers every table on the metadata
from backend.app.database.base import DB_SCHEMA

logger = logging.getLogger(__name__)


def create_db_engine(settings: Settings) -> Engine:
    """Create the engine with pooling configured from settings.

    * PostgreSQL: QueuePool (``DB_POOL_SIZE`` / ``DB_MAX_OVERFLOW`` / ``DB_POOL_TIMEOUT`` /
      ``DB_POOL_RECYCLE``) with ``pool_pre_ping`` so dead connections are replaced.
    * SQLite (dev/tests): the ``credit_risk`` schema is mapped to the default schema, foreign
      keys are enforced, and in-memory databases share a single connection.
    * ``hide_parameters=True`` keeps bound values (customer data) out of error messages/logs.
    """
    url = settings.sqlalchemy_url
    if settings.is_sqlite:
        kwargs: dict = {"connect_args": {"check_same_thread": False}}
        if url in ("sqlite://", "sqlite:///:memory:"):
            kwargs["poolclass"] = StaticPool
        engine = create_engine(url, hide_parameters=True, **kwargs)

        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _record) -> None:  # noqa: ANN001
            dbapi_connection.execute("PRAGMA foreign_keys=ON")

        engine = engine.execution_options(schema_translate_map={DB_SCHEMA: None})
    else:
        engine = create_engine(
            url,
            hide_parameters=True,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout,
            pool_recycle=settings.db_pool_recycle,
            pool_pre_ping=True,
        )
    logger.info("database_engine_created", extra={"dialect": engine.dialect.name})
    return engine


def init_db(engine: Engine) -> None:
    """Create the schema (PostgreSQL) and any missing tables. Idempotent."""
    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {DB_SCHEMA}"))
    models.Base.metadata.create_all(engine)
    logger.info("database_tables_ready")


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Session factory (``expire_on_commit=False`` so returned objects stay readable)."""
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Transactional scope for scripts / startup tasks (commit on success, rollback on error)."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
