"""Liveness and readiness probes."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Response
from sqlalchemy import text

from atlas_api.deps import EngineDep, RedisDep
from atlas_api.schemas import ReadyOut

router = APIRouter(tags=["health"])
_log = structlog.get_logger(__name__)


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz", response_model=ReadyOut)
async def readyz(response: Response, engine: EngineDep, redis: RedisDep) -> ReadyOut:
    database = True
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 — readiness must never raise
        database = False
        _log.warning("readyz.database", error=str(exc))

    redis_ok = True
    try:
        await redis.ping()
    except Exception as exc:  # noqa: BLE001
        redis_ok = False
        _log.warning("readyz.redis", error=str(exc))

    if not (database and redis_ok):
        response.status_code = 503
        return ReadyOut(status="degraded", database=database, redis=redis_ok)
    return ReadyOut(status="ok", database=True, redis=True)
