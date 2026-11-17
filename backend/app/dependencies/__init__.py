"""Dependency injection providers."""

from backend.app.dependencies.providers import (
    CustomerServiceDep,
    DashboardServiceDep,
    ModelServiceDep,
    PredictionServiceDep,
    SessionDep,
    SettingsDep,
    get_db,
)

__all__ = [
    "CustomerServiceDep",
    "DashboardServiceDep",
    "ModelServiceDep",
    "PredictionServiceDep",
    "SessionDep",
    "SettingsDep",
    "get_db",
]
