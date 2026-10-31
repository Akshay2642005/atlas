"""Both composition roots must be able to build the content store."""

from __future__ import annotations

from pathlib import Path

from atlas_core.config import AtlasSettings, StorageSettings
from atlas_core.domain.ports import ContentStorePort


def _settings(tmp_path: Path) -> AtlasSettings:
    return AtlasSettings(storage=StorageSettings(backend="local", local_root=str(tmp_path)))


async def test_api_container_resolves_the_content_store(tmp_path: Path) -> None:
    from atlas_api.container import build_container

    container = build_container(_settings(tmp_path))
    try:
        store = await container.get(ContentStorePort)
    finally:
        await container.close()

    assert isinstance(store, ContentStorePort)


async def test_worker_container_resolves_the_content_store(tmp_path: Path) -> None:
    from atlas_worker.container import build_container

    container = build_container(_settings(tmp_path))
    try:
        store = await container.get(ContentStorePort)
    finally:
        await container.close()

    assert isinstance(store, ContentStorePort)
