"""Dishka provider for storage adapters."""

from __future__ import annotations

from dishka import Provider, Scope, provide
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from atlas_core.config import AtlasSettings
from atlas_core.domain.ports import DocumentRepositoryPort, VectorRepositoryPort

SessionFactory = async_sessionmaker[AsyncSession]


class StorageProvider(Provider):
    @provide(scope=Scope.APP)
    def engine(self, settings: AtlasSettings) -> AsyncEngine:
        from atlas_vector.engine import create_engine_and_session

        engine, _ = create_engine_and_session(settings.postgres, instrument=True)
        return engine

    @provide(scope=Scope.APP)
    def session_factory(self, engine: AsyncEngine) -> SessionFactory:
        return async_sessionmaker(engine, expire_on_commit=False)

    @provide(scope=Scope.APP)
    def vector_repository(self, session_factory: SessionFactory) -> VectorRepositoryPort:
        from atlas_vector.repositories import PgVectorRepository

        return PgVectorRepository(session_factory)

    @provide(scope=Scope.APP)
    def document_repository(self, session_factory: SessionFactory) -> DocumentRepositoryPort:
        from atlas_vector.repositories import PgDocumentRepository

        return PgDocumentRepository(session_factory)


def providers() -> list[Provider]:
    return [StorageProvider()]
