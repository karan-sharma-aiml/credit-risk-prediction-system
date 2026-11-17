"""Aggregate router."""

from fastapi import APIRouter

from backend.app.api.routers import customers, dashboard, health, model_info, predict, predictions

api_router = APIRouter()
for module in (health, model_info, predict, customers, predictions, dashboard):
    api_router.include_router(module.router)
