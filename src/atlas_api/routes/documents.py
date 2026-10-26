"""Document ingestion endpoints (upload -> queue -> worker pipeline)."""

from __future__ import annotations

import base64
import binascii
import uuid

from fastapi import APIRouter, HTTPException, status
from sqlalchemy.exc import IntegrityError

from atlas_api.deps import Authenticated, DocumentsDep, QueueDep
from atlas_api.schemas import DocumentAcceptedOut, DocumentIn, DocumentOut
from atlas_core.domain.models import Document, DocumentStatus
from atlas_ingest.loaders import checksum

router = APIRouter(prefix="/v1/documents", tags=["documents"])


def _decode(content: str, encoding: str) -> bytes:
    if encoding == "utf8":
        return content.encode("utf-8")
    try:
        return base64.b64decode(content, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="invalid base64") from exc


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=DocumentAcceptedOut)
async def upload(
    body: DocumentIn,
    tenant: Authenticated,
    documents: DocumentsDep,
    queue: QueueDep,
) -> DocumentAcceptedOut:
    raw = _decode(body.content, body.content_encoding)
    document = Document(
        id=uuid.uuid4(),
        tenant_id=tenant.tenant_id,
        source_uri=body.source_uri,
        mime=body.mime,
        checksum=checksum(raw),
    )
    try:
        await documents.add(document, raw)
    except IntegrityError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="document already ingested") from exc
    job_id = await queue.enqueue_ingest(tenant.tenant_id, document.id)
    return DocumentAcceptedOut(id=document.id, status=DocumentStatus.PENDING, job_id=job_id)


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


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    tenant: Authenticated,
    documents: DocumentsDep,
) -> None:
    deleted = await documents.delete(tenant.tenant_id, document_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document not found")
