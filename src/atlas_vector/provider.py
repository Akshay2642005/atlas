"""Dishka provider for storage adapters."""

from __future__ import annotations

from dishka import Provider, Scope, provide
from sqlalchemy.ext.asyncio import AsyncEngine

from atlas_core.config import AtlasSettings
from atlas_vector.engine import SessionFactory, create_engine_and_session
from atlas_vector.repositories import PgDocumentRepository, PgVectorRepository


class StorageProvider(Provider):
    @provide(scope=Scope.APP)
    def engine_and_session(self, settings: AtlasSettings) -> tuple[AsyncEngine, SessionFactory]:
        return create_engine_and_session(settings.postgres, instrument=True)

    @provide(scope=Scope.APP)
    def vector_repository(
        self, engine_and_session: tuple[AsyncEngine, SessionFactory]
    ) -> PgVectorRepository:
        return PgVectorRepository(engine_and_session[1])

    @provide(scope=Scope.APP)
    def document_repository(
        self, engine_and_session: tuple[AsyncEngine, SessionFactory]
    ) -> PgDocumentRepository:
        return PgDocumentRepository(engine_and_session[1])


def providers() -> list[Provider]:
    return [StorageProvider()]
