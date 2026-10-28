"""Ports (hexagonal interfaces). Adapters live in other packages."""

from __future__ import annotations

import enum
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from atlas_core.domain.models import (
    ChatMessage,
    Chunk,
    Document,
    DocumentStatus,
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
    async def add(self, document: Document, content: bytes) -> None:
        """Persist a document and its raw content.

        Re-adding the same `(tenant_id, checksum)` replaces an earlier attempt
        that produced no chunks, so a failed ingest is retryable. A document
        that already produced chunks still conflicts.
        """
        ...

    async def get(self, tenant_id: str, document_id: uuid.UUID) -> Document | None: ...

    async def load_content(self, tenant_id: str, document_id: uuid.UUID) -> bytes | None: ...

    async def set_status(
        self, tenant_id: str, document_id: uuid.UUID, status: DocumentStatus, *, error: str | None = None
    ) -> None: ...

    async def list_latest(self, tenant_id: str, *, limit: int) -> list[Document]: ...

    async def delete(self, tenant_id: str, document_id: uuid.UUID) -> bool: ...


@runtime_checkable
class VectorRepositoryPort(Protocol):
    async def upsert_chunks(
        self, tenant_id: str, document_id: uuid.UUID, chunks: Sequence[tuple[Chunk, Sequence[float]]]
    ) -> int: ...

    async def semantic_search(
        self, tenant_id: str, embedding: Sequence[float], *, k: int
    ) -> list[tuple[Chunk, float]]: ...

    async def lexical_search(self, tenant_id: str, text: str, *, k: int) -> list[tuple[Chunk, float]]: ...


@runtime_checkable
class QueuePort(Protocol):
    async def enqueue_ingest(self, tenant_id: str, document_id: uuid.UUID) -> str: ...


@runtime_checkable
class PiiDetectorPort(Protocol):
    def detect(self, text: str) -> list[PiiFinding]: ...


@runtime_checkable
class InjectionGuardPort(Protocol):
    def scan(self, text: str) -> InjectionVerdict: ...
