"""arq task definitions for ingestion."""

from __future__ import annotations

import uuid
from typing import Any

import structlog

from atlas_core.observability import get_tracer
from atlas_ingest.pipeline import IngestPipeline

_log = structlog.get_logger(__name__)
_tracer = get_tracer("atlas.worker")


async def ingest_document(ctx: dict[str, Any], tenant_id: str, document_id: str) -> dict[str, Any]:
    """Idempotent-by-status job: process a pending document end to end."""
    container = ctx["container"]
    doc_uuid = uuid.UUID(document_id)
    with _tracer.start_as_current_span(
        "worker.ingest", attributes={"tenant.id": tenant_id, "document.id": document_id}
    ):
        pipeline: IngestPipeline = await container.get(IngestPipeline)
        result = await pipeline.process_document(tenant_id, doc_uuid)
        _log.info("worker.ingest.done", document_id=document_id, chunks=result.chunks)
        return {"document_id": document_id, "chunks": result.chunks}
