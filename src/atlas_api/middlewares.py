"""Auth + tenant resolution, rate limiting, error envelope, HTTP middlewares."""

from __future__ import annotations

import re
import time
import uuid
from typing import Annotated

import structlog
from dishka.integrations.fastapi import FromDishka
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from redis.asyncio import Redis

from atlas_core.config import AtlasSettings
from atlas_core.domain.errors import AtlasError, RateLimitExceeded


class TenantKey:
    """Resolved authentication context: which tenant is calling, via which key."""

    __slots__ = ("tenant_id", "key_prefix")

    def __init__(self, tenant_id: str, key_prefix: str) -> None:
        self.tenant_id = tenant_id
        self.key_prefix = key_prefix

    def rate_key(self) -> str:
        return f"rl:{self.key_prefix}"


_BEARER_RE = re.compile(r"^Bearer\s+(.+)$", re.IGNORECASE)


def _extract_api_key(request: Request) -> str:
    header = request.headers.get("X-API-Key", "")
    if header:
        return header.strip()
    authz = request.headers.get("Authorization", "")
    if match := _BEARER_RE.match(authz):
        return match.group(1).strip()
    return ""


def require_tenant(request: Request, settings: Annotated[AtlasSettings, FromDishka]) -> TenantKey:
    api_key = _extract_api_key(request)
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing API key",
            headers={"WWW-Authenticate": "ApiKey Bearer"},
        )
    tenant = settings.security.api_key_map().get(api_key)
    if tenant is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid API key")
    return TenantKey(tenant, api_key[:8])


async def enforce_rate_limit(
    tenant_key: TenantKey,
    redis: Annotated[Redis, FromDishka],
    settings: Annotated[AtlasSettings, FromDishka],
) -> None:
    """Sliding-window counter keyed by API-key prefix; raises typed 429 errors."""
    window = 60
    bucket = f"{tenant_key.rate_key()}:{int(time.time() // window)}"
    pipe = redis.pipeline(transaction=False)
    pipe.incr(bucket)
    pipe.expire(bucket, window * 2)
    count, _ = await pipe.execute()
    if int(count) > settings.security.rate_limit_rpm:
        raise RateLimitExceeded("rate limit exceeded", retry_after=1.0)


async def request_id_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(request_id=rid)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response
    finally:
        structlog.contextvars.unbind_contextvars()


async def security_headers_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    response =  await call_next(request)
    for header, value in (
        ("X-Content-Type-Options", "nosniff"),
        ("X-Frame-Options", "DENY"),
        ("Referrer-Policy", "no-referrer"),
        ("Cache-Control", "no-store"),
    ):
        response.headers.setdefault(header, value)
    return response


def error_envelope_handler(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, RateLimitExceeded):
        return JSONResponse(
            status_code=429,
            content={"error": exc.to_payload()},
            headers={"Retry-After": str(int(exc.retry_after) + 1)},
        )
    if isinstance(exc, AtlasError):
        return JSONResponse(status_code=exc.status, content={"error": exc.to_payload()})
    structlog.get_logger(__name__).error("unhandled_error", error=repr(exc))
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "internal_error", "message": "internal server error", "details": {}}},
    )
