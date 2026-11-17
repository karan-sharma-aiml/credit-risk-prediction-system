"""Structured (JSON) logging with sensitive-data protection.

Rules enforced here:

* Customer identifiers are logged only as keyed hashes (:func:`pseudonymize`).
* Any ``extra`` field whose name is a known sensitive key is replaced by
  ``[REDACTED]`` (:data:`SENSITIVE_KEYS`).
* Request bodies, SQL parameters and validation inputs are never logged
  (the DB engine is created with ``hide_parameters=True``).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from ml.config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET_COL

request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)

SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        *NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS, TARGET_COL, "customer_id",
        "default_probability", "input_snapshot", "payload", "body", "password",
        "secret", "secret_key", "token", "authorization", "database_url",
    }
)
_STANDARD_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime"}


def safe_path(request: Any) -> str:
    """Route template (``/customers/{customer_id}``) instead of the raw URL path.

    Raw paths contain customer identifiers, so they must never be logged.
    """
    route = request.scope.get("route")
    return getattr(route, "path", "unmatched")


def pseudonymize(value: object, secret: str) -> str:
    """Return a stable, non-reversible token for ``value`` (HMAC-SHA256, 12 hex chars)."""
    digest = hmac.new(secret.encode(), str(value).encode(), hashlib.sha256).hexdigest()
    return digest[:12]


class JsonFormatter(logging.Formatter):
    """Render each record as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = request_id_ctx.get()
        if request_id:
            payload["request_id"] = request_id
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = "[REDACTED]" if key.lower() in SENSITIVE_KEYS else value
        if record.exc_info and record.exc_info[0] is not None:
            # Exception type + traceback frames. Messages from our own code never embed customer
            # values and the DB layer hides statement parameters.
            payload["exc_type"] = record.exc_info[0].__name__
            payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable format for local development."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Configure the root logger once (idempotent: replaces existing handlers)."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_output else TextFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())
    for noisy in ("matplotlib", "multipart", "httpcore", "httpx", "httpx2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
