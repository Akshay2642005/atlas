"""Atlas API entry point."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from atlas_api.middlewares import (
    error_envelope_handler,
    request_id_middleware,
    security_headers_middleware,
)
from atlas_core.config import load_settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Atlas RAG API")

    # Add middlewares
    app.middleware("http")(request_id_middleware)
    app.middleware("http")(security_headers_middleware)
    app.add_exception_handler(Exception, error_envelope_handler)

    return app


app = create_app()


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}

def run() -> None:
    """Run the API server."""
    settings = load_settings()
    uvicorn.run(app, host=settings.api.host, port=settings.api.port)


if __name__ == "__main__":
    run()
