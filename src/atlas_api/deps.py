"""FastAPI dependency adapters over the Dishka container.

We intentionally do not use dishka's FastAPI route integration: it targets older
FastAPI internals. Dependencies are resolved from `app.state.container`, which
keeps request handling independent of framework-version churn.
"""

from __future__ import annotations

from typing import Annotated, Any, cast

from dishka import AsyncContainer
from fastapi import Depends, Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine

from atlas_api.middlewares import TenantKey, authenticated
from atlas_core.config import AtlasSettings
from atlas_core.domain.ports import DocumentRepositoryPort, QueuePort
from atlas_ingest.pipeline import IngestPipeline
from atlas_retrieval.answerer import AnswerService


async def _resolve(request: Request, tp: Any) -> Any:
    container = cast(AsyncContainer, request.app.state.container)
    return await container.get(tp)


async def get_settings(request: Request) -> AtlasSettings:
    return cast(AtlasSettings, await _resolve(request, AtlasSettings))


async def get_answer_service(request: Request) -> AnswerService:
    return cast(AnswerService, await _resolve(request, AnswerService))


async def get_documents(request: Request) -> DocumentRepositoryPort:
    return cast(DocumentRepositoryPort, await _resolve(request, DocumentRepositoryPort))


async def get_pipeline(request: Request) -> IngestPipeline:
    return cast(IngestPipeline, await _resolve(request, IngestPipeline))


async def get_queue(request: Request) -> QueuePort:
    return cast(QueuePort, await _resolve(request, QueuePort))


async def get_engine(request: Request) -> AsyncEngine:
    return cast(AsyncEngine, await _resolve(request, AsyncEngine))


async def get_redis(request: Request) -> Redis:
    return cast(Redis, await _resolve(request, Redis))


SettingsDep = Annotated[AtlasSettings, Depends(get_settings)]
AnswerServiceDep = Annotated[AnswerService, Depends(get_answer_service)]
DocumentsDep = Annotated[DocumentRepositoryPort, Depends(get_documents)]
QueueDep = Annotated[QueuePort, Depends(get_queue)]
PipelineDep = Annotated[IngestPipeline, Depends(get_pipeline)]
EngineDep = Annotated[AsyncEngine, Depends(get_engine)]
RedisDep = Annotated[Redis, Depends(get_redis)]
Authenticated = Annotated[TenantKey, Depends(authenticated)]

__all__ = [
    "AnswerServiceDep",
    "Authenticated",
    "DocumentsDep",
    "EngineDep",
    "PipelineDep",
    "QueueDep",
    "RedisDep",
    "SettingsDep",
    "TenantKey",
]
