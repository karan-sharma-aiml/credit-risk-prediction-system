"""Request-ID propagation and structured access logging."""

from __future__ import annotations

import logging
import re
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from backend.app.core.logging import request_id_ctx, safe_path

logger = logging.getLogger("api.access")
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
REQUEST_ID_HEADER = "X-Request-ID"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Assign a request id, time the request and log one line per request.

    Only the method, route template (never the raw path, which can contain a
    customer id), status and duration are logged. Query strings, headers and
    bodies are never logged.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = incoming if _SAFE_ID.match(incoming) else uuid.uuid4().hex
        token = request_id_ctx.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("request_crashed", extra={"method": request.method, "path": safe_path(request)})
            raise
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.info(
            "request_completed",
            extra={
                "method": request.method,
                "path": safe_path(request),
                "status": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        response.headers[REQUEST_ID_HEADER] = request_id
        request_id_ctx.reset(token)
        return response
