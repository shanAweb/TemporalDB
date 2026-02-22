"""
app/api/middleware.py

Custom ASGI middleware for TemporalDB.

RequestLoggingMiddleware
    Logs every HTTP request with method, path, status code, and wall-clock
    duration using structlog structured JSON.  Excluded from logging:
      - GET /health  (high-frequency liveness probe)
      - GET /docs, /redoc, /openapi.json  (OpenAPI UI assets)
"""
from __future__ import annotations

import time

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger(__name__)

# Paths that generate too much noise to log on every call.
_SILENT_PATHS: frozenset[str] = frozenset(
    {"/health", "/docs", "/redoc", "/openapi.json"}
)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log each request's method, path, status code, and latency."""

    async def dispatch(self, request: Request, call_next: object) -> Response:
        start = time.perf_counter()
        response: Response = await call_next(request)  # type: ignore[arg-type]
        duration_ms = (time.perf_counter() - start) * 1000

        if request.url.path not in _SILENT_PATHS:
            logger.info(
                "http_request",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
                client=request.client.host if request.client else None,
            )

        return response
