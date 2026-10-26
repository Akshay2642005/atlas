"""arq worker entrypoint: `uv run arq atlas_worker.main.WorkerSettings`."""

from __future__ import annotations

from typing import Any

from arq.connections import RedisSettings
from dishka import AsyncContainer

from atlas_core.config import load_settings
from atlas_worker.container import build_container
from atlas_worker.tasks import ingest_document


async def startup(ctx: dict[str, Any]) -> None:
    container: AsyncContainer = build_container()
    await container.__aenter__()
    ctx["container"] = container


async def shutdown(ctx: dict[str, Any]) -> None:
    container: AsyncContainer | None = ctx.get("container")
    if container is not None:
        await container.close()


class WorkerSettings:
    functions = [ingest_document]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(load_settings().redis.url)
    max_jobs = 4
    job_timeout = 600
    max_tries = 3
