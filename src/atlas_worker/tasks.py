"""arq task definitions for ingestion."""

from __future__ import annotations

import uuid
from typing import Any

import structlog

from atlas_core.observability import get_tracer
from atlas_ingest.pipeline import IngestPipeline

_log = structlog.get_logger(__name__)
_tracer = get_tracer("atlas.worker")


async def ingest_document(ctx: dict[str, Any], tenant_id: str, version_id: str) -> dict[str, Any]:
    """Process one document version. Retrying is safe: the work is per version."""
    container = ctx["container"]
    version_uuid = uuid.UUID(version_id)
    with _tracer.start_as_current_span(
        "worker.ingest", attributes={"tenant.id": tenant_id, "version.id": version_id}
    ):
        pipeline: IngestPipeline = await container.get(IngestPipeline)
        result = await pipeline.process_version(tenant_id, version_uuid)
        _log.info(
            "worker.ingest.done",
            document_id=str(result.document.id),
            version_id=version_id,
            chunks=result.chunks,
        )
        return {"document_id": str(result.document.id), "version_id": version_id, "chunks": result.chunks}
