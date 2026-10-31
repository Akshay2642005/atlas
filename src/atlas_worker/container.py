"""Worker DI composition root (no HTTP/queue-producer wiring needed)."""

from __future__ import annotations

from dishka import AsyncContainer, make_async_container

from atlas_core.config import AtlasSettings
from atlas_core.di import providers as core_providers
from atlas_ingest import providers as ingest_providers
from atlas_llm import providers as llm_providers
from atlas_storage import providers as storage_providers
from atlas_vector import providers as vector_providers


def build_container(settings: AtlasSettings | None = None) -> AsyncContainer:
    return make_async_container(
        *core_providers(settings),
        *llm_providers(),
        *vector_providers(),
        *storage_providers(),
        *ingest_providers(),
    )
