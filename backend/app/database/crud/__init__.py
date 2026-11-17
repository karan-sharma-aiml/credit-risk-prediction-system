"""Repositories (the CRUD layer)."""

from backend.app.database.crud.customers import CustomerRepository
from backend.app.database.crud.models import ModelRegistryRepository
from backend.app.database.crud.predictions import PredictionRepository

__all__ = ["CustomerRepository", "ModelRegistryRepository", "PredictionRepository"]
