"""Domain exceptions and the HTTP error contract.

Every error response has the shape::

    {"error": {"code": "...", "message": "...", "request_id": "...", "details": [...]}}

Validation details contain the field and the message only; the offending
input value is deliberately NOT echoed back or logged.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.core.logging import request_id_ctx, safe_path

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base class for expected, client-facing errors."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ConflictError(AppError):
    status_code = 409
    code = "conflict"


class ModelNotLoadedError(AppError):
    status_code = 503
    code = "model_unavailable"


class PredictionError(AppError):
    status_code = 500
    code = "prediction_failed"


def error_body(code: str, message: str, details: list[dict] | None = None) -> dict:
    """Build the standard error envelope."""
    body: dict = {"code": code, "message": message, "request_id": request_id_ctx.get()}
    if details:
        body["details"] = details
    return {"error": body}


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    logger.log(
        logging.ERROR if exc.status_code >= 500 else logging.INFO,
        "request_failed",
        extra={"error_code": exc.code, "status": exc.status_code, "path": safe_path(request)},
    )
    return JSONResponse(error_body(exc.code, exc.message), status_code=exc.status_code)


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    details = [
        {
            "field": ".".join(str(p) for p in err["loc"] if p != "body"),
            "message": err["msg"],
            "type": err["type"],
        }
        for err in exc.errors()
    ]
    logger.info("validation_failed", extra={"path": safe_path(request), "error_count": len(details)})
    return JSONResponse(error_body("validation_error", "Request validation failed.", details), status_code=422)


async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return JSONResponse(error_body("http_error", str(exc.detail)), status_code=exc.status_code)


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_exception", extra={"path": safe_path(request)})
    return JSONResponse(error_body("internal_error", "An unexpected error occurred."), status_code=500)


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all handlers to ``app``."""
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
