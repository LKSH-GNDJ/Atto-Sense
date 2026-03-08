"""
BotTrain v3.1 - Structured Logging
JSON-formatted logs with per-request context (session_id, modality, latency).
Compatible with Datadog, Grafana Loki, CloudWatch, and any OTLP log pipeline.

Usage:
    from backend.core.logging_config import get_logger
    log = get_logger(__name__)
    log.info("classified", extra={"session_id": sid, "intent": intent, "latency_ms": 123.4})
"""

import os
import sys
import logging
import time
from typing import Callable

from pythonjsonlogger import jsonlogger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# ── Log Level ──────────────────────────────────────────────────────────────────

_LEVEL_MAP = {
    "debug":    logging.DEBUG,
    "info":     logging.INFO,
    "warning":  logging.WARNING,
    "error":    logging.ERROR,
    "critical": logging.CRITICAL,
}
LOG_LEVEL = _LEVEL_MAP.get(os.getenv("LOG_LEVEL", "info").lower(), logging.INFO)


# ── JSON Formatter ─────────────────────────────────────────────────────────────

class _BotTrainFormatter(jsonlogger.JsonFormatter):
    """
    Extends python-json-logger with fixed BotTrain fields:
      service, environment, version
    """
    _SERVICE = "bottrain-api"
    _ENV     = os.getenv("ENVIRONMENT", "development")
    _VERSION = "3.1.1"

    def add_fields(self, log_record: dict, record: logging.LogRecord, message_dict: dict):
        super().add_fields(log_record, record, message_dict)
        log_record["service"]     = self._SERVICE
        log_record["environment"] = self._ENV
        log_record["version"]     = self._VERSION
        log_record["level"]       = record.levelname
        # ISO timestamp
        log_record["timestamp"]   = self.formatTime(record, self.datefmt)
        # Remove redundant fields added by base class
        log_record.pop("asctime", None)


# ── Root Logger Setup ──────────────────────────────────────────────────────────

def configure_logging() -> None:
    """Call once at application startup."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_BotTrainFormatter(
        fmt="%(timestamp)s %(level)s %(name)s %(message)s"
    ))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(LOG_LEVEL)

    # Quiet noisy third-party loggers
    for noisy in ("uvicorn.access", "sqlalchemy.engine", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# ── Request Logging Middleware ─────────────────────────────────────────────────

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs one JSON line per request with:
      method, path, status_code, duration_ms, client_ip, session_id (if present)
    """

    def __init__(self, app, logger: logging.Logger | None = None):
        super().__init__(app)
        self._log = logger or get_logger("bottrain.http")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        t0 = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - t0) * 1000

        # Don't spam logs with health-check noise
        if request.url.path == "/health":
            return response

        self._log.info(
            "request",
            extra={
                "method":      request.method,
                "path":        request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
                "client_ip":   request.client.host if request.client else "unknown",
            },
        )
        return response
