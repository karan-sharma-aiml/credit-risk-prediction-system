"""Dashboard configuration (environment driven)."""

from __future__ import annotations

import os

DEFAULT_API_URL = "http://localhost:8000"

RISK_COLORS = {"low": "#2E9E6B", "medium": "#F2A93B", "high": "#D64545"}
RISK_LABELS = {"low": "Low Risk", "medium": "Medium Risk", "high": "High Risk"}


def normalize_base_url(url: str) -> str:
    """Add a scheme if missing (Render passes ``host:port``) and strip trailing slashes."""
    url = url.strip().rstrip("/")
    return url if url.startswith(("http://", "https://")) else f"http://{url}"


def api_url() -> str:
    """Base URL of the FastAPI backend (``API_URL`` environment variable)."""
    return normalize_base_url(os.getenv("API_URL", DEFAULT_API_URL))
