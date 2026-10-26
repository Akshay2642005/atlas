"""Alembic environment: async engine driven by AtlasSettings."""

from __future__ import annotations

import asyncio

from alembic import context

from atlas_core.config import load_settings
from atlas_vector.schema import Base

config = context.config
target_metadata = Base.metadata


def do_run(connection) -> None:  # type: ignore[no-untyped-def]
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_async() -> None:
    from atlas_vector.engine import create_engine_and_session

    settings = load_settings()
    engine, _ = create_engine_and_session(settings.postgres)
    async with engine.connect() as connection:
        await connection.run_sync(do_run)
    await engine.dispose()


if context.is_offline_mode():
    settings = load_settings()
    context.configure(url=settings.postgres.url, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
else:
    asyncio.run(run_async())
