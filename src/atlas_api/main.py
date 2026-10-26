"""Atlas API entry point."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from atlas_api.container import build_container
from atlas_api.middlewares import (
    error_envelope_handler,
    request_id_middleware,
    security_headers_middleware,
)
from atlas_api.routes import chat_router, documents_router, health_router
from atlas_core.config import AtlasSettings, load_settings
from atlas_core.domain.errors import AtlasError
from atlas_core.observability import setup_logging, setup_tracing


def create_app(settings: AtlasSettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    s = settings or load_settings()
    setup_logging(s.observability)
    tracer_provider = setup_tracing(s.observability)

    container = build_container(s)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        yield
        await container.close()

    app = FastAPI(
        title="Atlas RAG API",
        version="0.1.0",
        description="Tenant-isolated RAG: ingestion, hybrid retrieval, grounded answers.",
        lifespan=lifespan,
    )
    app.state.container = container

    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(documents_router)

    app.add_exception_handler(AtlasError, error_envelope_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, error_envelope_handler)

    # Outermost registered last.
    app.middleware("http")(request_id_middleware)
    app.middleware("http")(security_headers_middleware)

    if tracer_provider is not None:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider)
    return app


app = create_app()


def run() -> None:
    """Run the API server (local dev / container entrypoint)."""
    settings = load_settings()
    uvicorn.run("atlas_api.main:app", host=settings.api.host, port=settings.api.port)


if __name__ == "__main__":
    run()
