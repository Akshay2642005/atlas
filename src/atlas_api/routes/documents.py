"""Document ingestion endpoints (upload -> store -> commit -> queue)."""

from __future__ import annotations

import base64
import binascii
import uuid

from fastapi import APIRouter, HTTPException, Security, status

from atlas_api.deps import Authenticated, DocumentsDep, PipelineDep, QueueDep
from atlas_api.schemas import DocumentAcceptedOut, DocumentIn, DocumentOut, DocumentVersionOut
from atlas_api.security import API_KEY_SCHEME, AUTH_RESPONSES, BEARER_SCHEME
from atlas_core.domain.models import DocumentStatus
from atlas_ingest.pipeline import IngestSpec

# The `Security(...)` entries are documentation: they register the schemes so
# Swagger can send a key. `auto_error=False` keeps enforcement in
# `middlewares.authenticated`, which reads the same two headers.
router = APIRouter(
    prefix="/v1/documents",
    tags=["documents"],
    dependencies=[Security(API_KEY_SCHEME), Security(BEARER_SCHEME)],
    responses=AUTH_RESPONSES,
)


def _decode(content: str, encoding: str) -> bytes:
    if encoding == "utf8":
        return content.encode("utf-8")
    try:
        return base64.b64decode(content, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="invalid base64"
        ) from exc


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=DocumentAcceptedOut)
async def upload(
    body: DocumentIn,
    tenant: Authenticated,
    pipeline: PipelineDep,
    queue: QueueDep,
) -> DocumentAcceptedOut:
    """Store the bytes, commit a version, and queue it.

    Re-uploading the head's exact bytes returns `created=false` and queues
    nothing: the version is already serving. Different bytes append a new
    version, which is the edit path — the old version stays searchable until the
    new one is indexed.
    """
    committed = await pipeline.store(
        IngestSpec(
            tenant_id=tenant.tenant_id,
            content=_decode(body.content, body.content_encoding),
            mime=body.mime,
            source_uri=body.source_uri,
        )
    )
    job_id = ""
    if committed.version.status is not DocumentStatus.INDEXED:
        job_id = await queue.enqueue_ingest(tenant.tenant_id, committed.version.id)
    return DocumentAcceptedOut(
        id=committed.document.id,
        status=committed.version.status,
        version_id=committed.version.id,
        seq=committed.version.seq,
        created=committed.created,
        job_id=job_id,
    )


@router.get("", response_model=list[DocumentOut])
async def list_documents(
    tenant: Authenticated,
    documents: DocumentsDep,
    limit: int = 20,
) -> list[DocumentOut]:
    rows = await documents.list_latest(tenant.tenant_id, limit=max(1, min(limit, 100)))
    return [DocumentOut.from_domain(doc) for doc in rows]


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: uuid.UUID,
    tenant: Authenticated,
    documents: DocumentsDep,
) -> DocumentOut:
    doc = await documents.get(tenant.tenant_id, document_id)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document not found")
    return DocumentOut.from_domain(doc)


@router.get("/{document_id}/versions", response_model=list[DocumentVersionOut])
async def list_versions(
    document_id: uuid.UUID,
    tenant: Authenticated,
    documents: DocumentsDep,
    limit: int = 50,
) -> list[DocumentVersionOut]:
    """The append-only lineage, newest first."""
    if await documents.get(tenant.tenant_id, document_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document not found")
    versions = await documents.list_versions(
        tenant.tenant_id, document_id, limit=max(1, min(limit, 200))
    )
    return [DocumentVersionOut.from_domain(version) for version in versions]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    tenant: Authenticated,
    documents: DocumentsDep,
) -> None:
    deleted = await documents.delete(tenant.tenant_id, document_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document not found")
