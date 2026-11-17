"""Analytics endpoint consumed by the Streamlit dashboard."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.dependencies import DashboardServiceDep
from backend.app.schemas.dashboard import DashboardSummary

router = APIRouter(tags=["dashboard"])


@router.get("/dashboard", response_model=DashboardSummary, summary="Portfolio KPIs, distributions and trends")
def dashboard(service: DashboardServiceDep) -> DashboardSummary:
    return service.summary()
