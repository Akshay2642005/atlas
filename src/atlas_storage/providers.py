"""Dishka provider for the content store.

Named `ObjectStorageProvider` rather than `StorageProvider` because
`atlas_vector.provider.StorageProvider` already owns the database adapters and
the two names would collide in the composition roots.
"""

from __future__ import annotations

from dishka import Provider, Scope, provide

from atlas_core.config import AtlasSettings
from atlas_core.domain.ports import ContentStorePort


class ObjectStorageProvider(Provider):
    @provide(scope=Scope.APP)
    def content_store(self, settings: AtlasSettings) -> ContentStorePort:
        from atlas_storage.obstore_store import build_content_store

        return build_content_store(settings.storage)


def providers() -> list[Provider]:
    return [ObjectStorageProvider()]
