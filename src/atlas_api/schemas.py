"""HTTP request/response models (strict, extra-forbidden)."""

from __future__ import annotations

import datetime
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from atlas_core.domain.models import Answer, Document, DocumentStatus, DocumentVersion


class ChatIn(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1, max_length=8000)
    stream: bool = False


class CitationOut(BaseModel):
    document_id: uuid.UUID
    chunk_id: uuid.UUID
    seq: int
    score: float


class TraceStepOut(BaseModel):
    name: str
    detail: str


class AnswerOut(BaseModel):
    answer: str
    citations: list[CitationOut]
    trace: list[TraceStepOut]

    @classmethod
    def from_domain(cls, answer: Answer) -> AnswerOut:
        return cls(
            answer=answer.text,
            citations=[
                CitationOut(document_id=c.document_id, chunk_id=c.chunk_id, seq=c.seq, score=c.score)
                for c in answer.citations
            ],
            trace=[TraceStepOut(name=t.name, detail=t.detail) for t in answer.trace],
        )


class DocumentIn(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source_uri: str = Field(min_length=1, max_length=2000)
    mime: Literal["text/plain", "text/markdown", "text/html", "application/pdf"] = "text/plain"
    content: str = Field(min_length=1, max_length=10_000_000)
    content_encoding: Literal["utf8", "base64"] = "utf8"


class DocumentOut(BaseModel):
    id: uuid.UUID
    tenant_id: str
    source_uri: str
    mime: str
    status: DocumentStatus
    error: str | None = None
    created_at: datetime.datetime | None = None
    # What retrieval is serving. The newest version can be a failed attempt on
    # top of it, which is why this is not the same thing as `status`.
    head_version_id: uuid.UUID | None = None

    @classmethod
    def from_domain(cls, doc: Document) -> DocumentOut:
        return cls(
            id=doc.id,
            tenant_id=doc.tenant_id,
            source_uri=doc.source_uri,
            mime=doc.mime,
            status=doc.status,
            error=doc.error,
            created_at=doc.created_at,
            head_version_id=doc.head_version_id,
        )


class DocumentVersionOut(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    seq: int
    status: DocumentStatus
    checksum: str
    mime: str
    chunk_count: int
    message: str | None = None
    error: str | None = None
    parent_version_id: uuid.UUID | None = None
    created_at: datetime.datetime | None = None

    @classmethod
    def from_domain(cls, version: DocumentVersion) -> DocumentVersionOut:
        return cls(
            id=version.id,
            document_id=version.document_id,
            seq=version.seq,
            status=version.status,
            checksum=version.checksum,
            mime=version.mime,
            chunk_count=version.chunk_count,
            message=version.message,
            error=version.error,
            parent_version_id=version.parent_version_id,
            created_at=version.created_at,
        )


class DocumentAcceptedOut(BaseModel):
    id: uuid.UUID
    status: DocumentStatus
    version_id: uuid.UUID
    seq: int
    # False when the head already held these bytes: an upload that changed
    # nothing, answered 202 rather than 409.
    created: bool
    job_id: str


class ReadyOut(BaseModel):
    status: Literal["ok", "degraded"]
    database: bool
    redis: bool
