"""Identifier and time helpers."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone


def generate_customer_id() -> str:
    """Return a new customer id such as ``CUST3F9A1C0B7D2E`` (16 chars, fits VARCHAR(20))."""
    return f"CUST{uuid.uuid4().hex[:12].upper()}"


def ensure_utc(value: datetime) -> datetime:
    """Treat naive datetimes (SQLite) as UTC so API output is always timezone-aware."""
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
