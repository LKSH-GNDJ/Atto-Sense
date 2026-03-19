"""
AttoSense v3.1 - Authentication
API key validation via X-API-Key header or ?api_key= query param.

Setup:
  Set API_KEY=your-secret-key in .env (generate with: openssl rand -hex 32)
  Set API_KEY_DISABLED=true to bypass auth in local dev (never in prod).

Public routes (no key required):
  GET /health
  GET /docs
  GET /openapi.json
  GET /redoc
"""

import os
import hmac
import hashlib
from fastapi import HTTPException, Security, Request
from fastapi.security import APIKeyHeader, APIKeyQuery

_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_KEY_QUERY  = APIKeyQuery(name="api_key",    auto_error=False)

# Routes that never require authentication
PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc", "/favicon.ico"}


def _get_configured_key() -> str | None:
    return os.getenv("API_KEY")


def _constant_time_compare(a: str, b: str) -> bool:
    """Timing-safe string comparison to prevent timing attacks."""
    return hmac.compare_digest(
        hashlib.sha256(a.encode()).digest(),
        hashlib.sha256(b.encode()).digest(),
    )


async def require_api_key(
    request: Request,
    header_key: str | None = Security(_KEY_HEADER),
    query_key:  str | None = Security(_KEY_QUERY),
) -> str:
    """
    FastAPI dependency — inject into routes that require auth.
    Usage:  async def my_route(api_key: str = Depends(require_api_key))
    """
    # Bypass in dev mode (only when explicitly set — never default to True)
    if os.getenv("API_KEY_DISABLED", "false").lower() == "true":
        return "dev-bypass"

    configured = _get_configured_key()
    if not configured:
        raise HTTPException(
            status_code=503,
            detail="API_KEY is not configured on this server.",
        )

    provided = header_key or query_key
    if not provided:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Supply X-API-Key header or ?api_key= param.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not _constant_time_compare(provided, configured):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )

    return provided


class AuthMiddleware:
    """
    ASGI middleware that enforces API key on ALL non-public routes.
    This catches requests before they reach any route handler,
    providing a single enforcement point regardless of Depends() usage.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        # Always allow public paths through
        if path in PUBLIC_PATHS or path.startswith("/static"):
            await self.app(scope, receive, send)
            return

        # Skip auth in dev mode
        if os.getenv("API_KEY_DISABLED", "false").lower() == "true":
            await self.app(scope, receive, send)
            return

        configured = _get_configured_key()
        if not configured:
            await self.app(scope, receive, send)
            return

        # Extract key from headers
        headers = dict(scope.get("headers", []))
        key_from_header = headers.get(b"x-api-key", b"").decode()

        # Extract key from query string
        query_string = scope.get("query_string", b"").decode()
        key_from_query = ""
        for part in query_string.split("&"):
            if part.startswith("api_key="):
                key_from_query = part[len("api_key="):]
                break

        provided = key_from_header or key_from_query

        if not provided or not _constant_time_compare(provided, configured):
            from starlette.responses import JSONResponse
            response = JSONResponse(
                {"detail": "Invalid or missing API key.", "success": False},
                status_code=401,
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
