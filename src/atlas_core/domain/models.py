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
    # mime, status, checksum and error are derived from the version lineage
    # (head if there is one, else the newest version) rather than stored on the
    # document, so they cannot drift from it.
    mime: str
    status: DocumentStatus = DocumentStatus.PENDING
    checksum: str = ""
    created_at: datetime.datetime | None = None
    error: str | None = None
    head_version_id: uuid.UUID | None = None


@dataclass(frozen=True, slots=True)
class Blob:
    """Immutable bytes, content-addressed within one tenant."""

    sha256: str
    size_bytes: int
    storage_key: str


@dataclass(frozen=True, slots=True)
class DocumentVersion:
    """One immutable entry in a document's lineage.

    `status` is per version: a retry updates the version in place, and a failed
    version never becomes head, so it is never searchable.
    """

    id: uuid.UUID
    document_id: uuid.UUID
    seq: int
    checksum: str
    mime: str
    status: DocumentStatus
    chunk_count: int = 0
    parent_version_id: uuid.UUID | None = None
    message: str | None = None
    error: str | None = None
    created_at: datetime.datetime | None = None
    # The blob this version points at, joined in on read so the pipeline can
    # stream the bytes without a second lookup on its own port.
    storage_key: str = ""
    size_bytes: int = 0


@dataclass(frozen=True, slots=True)
class CommitResult:
    """Outcome of storing bytes as a document version.

    `created=False` means the head already held these exact bytes: the upload is
    a no-op rather than a 409 or a duplicate version.
    """

    document: Document
    version: DocumentVersion
    created: bool


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
    document_version_id: uuid.UUID | None = None


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
