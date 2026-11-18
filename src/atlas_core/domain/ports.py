"""Ports (hexagonal interfaces). Adapters live in other packages."""

from __future__ import annotations

import datetime
import enum
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from atlas_core.domain.models import (
    Blob,
    ChatMessage,
    Chunk,
    CommitResult,
    Document,
    DocumentStatus,
    DocumentVersion,
    StoredObject,
)


class EmbedKind(enum.StrEnum):
    PASSAGE = "passage"
    QUERY = "query"


@dataclass(frozen=True, slots=True)
class PiiFinding:
    kind: str
    start: int
    end: int
    value: str


@dataclass(frozen=True, slots=True)
class InjectionVerdict:
    risky: bool
    matches: tuple[str, ...]


@runtime_checkable
class ChatPort(Protocol):
    async def complete(self, messages: Sequence[ChatMessage], *, max_tokens: int | None = None) -> str: ...

    def stream(self, messages: Sequence[ChatMessage], *, max_tokens: int | None = None) -> AsyncIterator[str]: ...


@runtime_checkable
class EmbedderPort(Protocol):
    async def embed(self, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]: ...


@runtime_checkable
class RerankerPort(Protocol):
    async def rerank(self, query: str, documents: Sequence[str]) -> list[float]: ...


@runtime_checkable
class ChunkerPort(Protocol):
    async def chunk(self, text: str) -> list[tuple[int, str]]: ...


@runtime_checkable
class DocumentRepositoryPort(Protocol):
    """Document identity, lineage, and the one mutable pointer.

    A document is an identity (`tenant_id`, `source_uri`) plus an append-only
    lineage of versions. Bytes live in the object store; this port never sees
    them, which is what keeps the repository a metadata store.
    """

    async def commit(
        self,
        *,
        tenant_id: str,
        source_uri: str,
        blob: Blob,
        mime: str,
        message: str | None = None,
    ) -> CommitResult:
        """Record `blob` as the next version of `source_uri`.

        Same bytes as the head returns `created=False` and writes nothing; new
        bytes append `seq = max + 1`. The head pointer does not move here — it
        moves when the version is `indexed`, so an unfinished version is never
        searchable.
        """
        ...

    async def get(self, tenant_id: str, document_id: uuid.UUID) -> Document | None: ...

    async def get_version(
        self, tenant_id: str, version_id: uuid.UUID
    ) -> DocumentVersion | None: ...

    async def list_versions(
        self, tenant_id: str, document_id: uuid.UUID, *, limit: int
    ) -> list[DocumentVersion]: ...

    async def set_version_status(
        self,
        tenant_id: str,
        version_id: uuid.UUID,
        status: DocumentStatus,
        *,
        error: str | None = None,
        chunk_count: int | None = None,
    ) -> None: ...

    async def set_head(
        self, tenant_id: str, document_id: uuid.UUID, version_id: uuid.UUID
    ) -> None:
        """Move the head pointer. Only called once a version is `indexed`."""
        ...

    async def resolve_serving_version(
        self, document: Document
    ) -> DocumentVersion | None:
        """The version retrieval serves: the head, which is always indexed.

        A failed version never becomes head, so this needs no ancestor walk; it
        returns `None` only while a document has never had a successful ingest.
        """
        ...

    async def list_latest(self, tenant_id: str, *, limit: int) -> list[Document]: ...

    async def delete(self, tenant_id: str, document_id: uuid.UUID) -> bool: ...


@runtime_checkable
class ContentStorePort(Protocol):
    """Immutable bytes in an object store, addressed by key.

    Keys are opaque to callers; `atlas_storage.keys` derives them. The store is
    never asked to interpret a key, which keeps S3, GCS, and a local directory
    behind one contract.
    """

    async def put_bytes(self, key: str, data: bytes) -> StoredObject: ...

    async def put_stream(self, key: str, chunks: AsyncIterator[bytes]) -> StoredObject: ...

    async def open_stream(self, key: str) -> AsyncIterator[bytes]: ...

    async def head(self, key: str) -> StoredObject | None:
        """Return metadata, or `None` when the key does not exist."""
        ...

    async def copy(self, source: str, destination: str) -> None: ...

    async def delete(self, key: str) -> None:
        """Delete a key. Deleting a missing key is not an error."""
        ...

    async def presign_put(self, key: str, *, expires_in: datetime.timedelta) -> str: ...

    async def presign_get(self, key: str, *, expires_in: datetime.timedelta) -> str: ...


@runtime_checkable
class VectorRepositoryPort(Protocol):
    async def upsert_chunks(
        self,
        tenant_id: str,
        document_version_id: uuid.UUID,
        chunks: Sequence[tuple[Chunk, Sequence[float]]],
    ) -> int:
        """Persist chunks under one version. Chunks of other versions are untouched."""
        ...

    async def semantic_search(
        self, tenant_id: str, embedding: Sequence[float], *, k: int
    ) -> list[tuple[Chunk, float]]: ...

    async def lexical_search(self, tenant_id: str, text: str, *, k: int) -> list[tuple[Chunk, float]]: ...


@runtime_checkable
class QueuePort(Protocol):
    async def enqueue_ingest(self, tenant_id: str, version_id: uuid.UUID) -> str:
        """Queue ingestion of one document version, not of the document."""
        ...


@runtime_checkable
class PiiDetectorPort(Protocol):
    def detect(self, text: str) -> list[PiiFinding]: ...


@runtime_checkable
class InjectionGuardPort(Protocol):
    def scan(self, text: str) -> InjectionVerdict: ...
