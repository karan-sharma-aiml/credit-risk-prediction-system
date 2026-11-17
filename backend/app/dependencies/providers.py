"""FastAPI dependency providers (dependency injection).

All shared state lives on ``app.state`` (set in the lifespan / app factory), so
nothing is a module-level singleton and tests can inject their own engine,
settings and predictor.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from backend.app.core.config import Settings
from backend.app.core.exceptions import ModelNotLoadedError
from backend.app.core.logging import pseudonymize
from backend.app.services.customer_service import CustomerService
from backend.app.services.dashboard_service import DashboardService
from backend.app.services.model_service import ModelContext, ModelService
from backend.app.services.prediction_service import PredictionService
from backend.app.services.recommendation import RecommendationEngine


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_db(request: Request) -> Iterator[Session]:
    """One session per request; rolled back on error, always closed."""
    session = request.app.state.session_factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_model_context(request: Request) -> ModelContext:
    context: ModelContext | None = request.app.state.model_context
    if context is None:
        raise ModelNotLoadedError("The prediction model is not loaded.")
    return context


def get_model_service(request: Request) -> ModelService:
    service: ModelService | None = request.app.state.model_service
    if service is None:
        raise ModelNotLoadedError("The prediction model is not loaded.")
    return service


def get_pseudonymizer(settings: Annotated[Settings, Depends(get_settings)]) -> Callable[[str], str]:
    secret = settings.secret_key.get_secret_value()
    return lambda value: pseudonymize(value, secret)


def get_recommender(request: Request) -> RecommendationEngine:
    return request.app.state.recommender


SessionDep = Annotated[Session, Depends(get_db)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
ModelContextDep = Annotated[ModelContext, Depends(get_model_context)]
ModelServiceDep = Annotated[ModelService, Depends(get_model_service)]
PseudonymizerDep = Annotated[Callable[[str], str], Depends(get_pseudonymizer)]
RecommenderDep = Annotated[RecommendationEngine, Depends(get_recommender)]


def get_customer_service(session: SessionDep, pseudonymizer: PseudonymizerDep) -> CustomerService:
    return CustomerService(session, pseudonymizer)


def get_prediction_service(
    session: SessionDep,
    context: ModelContextDep,
    recommender: RecommenderDep,
    pseudonymizer: PseudonymizerDep,
) -> PredictionService:
    return PredictionService(session, context, recommender, pseudonymizer)


def get_dashboard_service(
    session: SessionDep,
    context: ModelContextDep,
    prediction_service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> DashboardService:
    return DashboardService(session, prediction_service, context)


CustomerServiceDep = Annotated[CustomerService, Depends(get_customer_service)]
PredictionServiceDep = Annotated[PredictionService, Depends(get_prediction_service)]
DashboardServiceDep = Annotated[DashboardService, Depends(get_dashboard_service)]
