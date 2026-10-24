"""Async engine + session factory (OTel-instrumented)."""

from __future__ import annotations

from typing import Final

from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from atlas_core.config import AtlasSettings, PostgresSettings

_SessionFactory = async_sessionmaker[AsyncSession]


def create_engine_and_session(
    init: PostgresSettings, *, settings: AtlasSettings | None = None, instrument: bool = False
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    engine: Final = create_async_engine(
        init.url,
        pool_size=init.pool_size,
        pool_pre_ping=True,
        connect_args={"server_settings": {"statement_timeout": str(init.statement_timeout_ms)}},
        poolclass=NullPool if settings is not None and settings.env == "test" else None,
    )
    if instrument:
        SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine)
    return engine, async_sessionmaker(engine, expire_on_commit=False)
