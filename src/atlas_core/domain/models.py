"""Domain entities. Frozen dataclasses — no framework code lives here."""

from __future__ import annotations

import datetime
import enum
import uuid
from dataclasses import dataclass
from typing import Literal


class DocumentStatus(enum.StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Document:
    id: uuid.UUID
    tenant_id: str
    source_uri: str
    mime: str
    status: DocumentStatus = DocumentStatus.PENDING
    checksum: str = ""
    created_at: datetime.datetime | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class StoredObject:
    """Metadata for a document's bytes in the content store."""

    key: str
    size_bytes: int
    etag: str | None = None


@dataclass(frozen=True, slots=True)
class Chunk:
    id: uuid.UUID
    document_id: uuid.UUID
    tenant_id: str
    seq: int
    level: int = 0
    text: str = ""
    token_count: int = 0


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    """Chunk produced by a chunker, before persistence and embedding."""

    text: str
    seq: int
    level: int = 0
    token_count: int = 0


@dataclass(frozen=True, slots=True)
class Query:
    id: uuid.UUID
    tenant_id: str
    text: str


@dataclass(frozen=True, slots=True)
class Citation:
    document_id: uuid.UUID
    chunk_id: uuid.UUID
    source_uri: str
    seq: int
    score: float


@dataclass(frozen=True, slots=True)
class TraceStep:
    """One reasoning/observability step of a request (planned → retrieved → ...)."""

    name: str
    detail: str


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    citations: tuple[Citation, ...]
    trace: tuple[TraceStep, ...]


ChatRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: ChatRole
    content: str
