"""DI composition root for the API: core + adapters + queue + retrieval."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from dishka import AsyncContainer, Provider, Scope, make_async_container, provide

from atlas_core.config import AtlasSettings
from atlas_core.di import providers as core_providers
from atlas_core.domain.ports import QueuePort
from atlas_ingest import providers as ingest_providers
from atlas_llm import providers as llm_providers
from atlas_retrieval import providers as retrieval_providers
from atlas_vector import providers as vector_providers


class _ArqQueue(QueuePort):
    def __init__(self, pool: ArqRedis) -> None:
        self._pool = pool

    async def enqueue_ingest(self, tenant_id: str, document_id: uuid.UUID) -> str:
        job = await self._pool.enqueue_job(
            "ingest_document", tenant_id=tenant_id, document_id=str(document_id)
        )
        return job.job_id


class ApiProvider(Provider):
    @provide(scope=Scope.APP)
    async def queue(self, settings: AtlasSettings) -> AsyncIterator[QueuePort]:
        pool = await create_pool(RedisSettings.from_dsn(settings.redis.url))
        yield _ArqQueue(pool)
        await pool.aclose()


def build_container(settings: AtlasSettings | None = None) -> AsyncContainer:
    return make_async_container(
        [
            *core_providers(settings),
            *llm_providers(),
            *vector_providers(),
            *ingest_providers(),
            *retrieval_providers(),
            ApiProvider(),
        ]
    )
