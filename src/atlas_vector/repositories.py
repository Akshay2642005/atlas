"""pgVector repositories implementing core ports. Tenant isolation on every query."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import ColumnElement, delete, exists, select, update
from sqlalchemy import func as safunc
from sqlalchemy.dialects.postgresql import insert as pg_insert

from atlas_core.config import AtlasSettings
from atlas_core.domain.models import Chunk, Document, DocumentStatus
from atlas_core.domain.ports import DocumentRepositoryPort, VectorRepositoryPort
from atlas_vector.engine import SessionFactory, create_engine_and_session
from atlas_vector.schema import ChunkRow, DocumentContentRow, DocumentRow

SessionFactoryAlias = SessionFactory


def _lexical_tsquery(text: str) -> ColumnElement[Any]:
    """OR the question's lexemes into a tsquery.

    `plainto_tsquery` ANDs every term, so a natural-language question matches a
    chunk only if that chunk contains all of them - which is nothing in
    practice, leaving the lexical branch empty and RRF with a single list. OR
    the lexemes instead so the branch contributes real candidates.

    `ts_rank_cd` then scores by how many of the terms a chunk covers.
    """
    lexemes = safunc.tsvector_to_array(safunc.to_tsvector("english", text))
    terms = safunc.array_to_string(lexemes, " | ")
    # A stopword-only input yields "", which `to_tsquery` rejects as invalid.
    return safunc.to_tsquery("english", safunc.coalesce(safunc.nullif(terms, ""), "atlasempty"))


def _chunk(row: ChunkRow) -> Chunk:
    return Chunk(
        id=row.id,
        document_id=row.document_id,
        tenant_id=row.tenant_id,
        seq=row.seq,
        level=row.level,
        text=row.text,
        token_count=row.token_count,
    )


def _doc(row: DocumentRow) -> Document:
    return Document(
        id=row.id,
        tenant_id=row.tenant_id,
        source_uri=row.source_uri,
        mime=row.mime,
        status=DocumentStatus(row.status),
        checksum=row.checksum,
        created_at=row.created_at,
        error=row.error,
    )


class PgVectorRepository(VectorRepositoryPort):
    def __init__(self, session_factory: SessionFactoryAlias | None = None, settings: AtlasSettings | None = None) -> None:
        self._session_factory = session_factory or create_engine_and_session(
            settings.postgres  # type: ignore[union-attr]
        )[1]

    async def upsert_chunks(
        self, tenant_id: str, document_id: uuid.UUID, chunks: Sequence[tuple[Chunk, Sequence[float]]]
    ) -> int:
        async with self._session_factory() as session, session.begin():
            for chunk, embedding in chunks:
                stmt = pg_insert(ChunkRow).values(
                    id=chunk.id,
                    document_id=chunk.document_id,
                    tenant_id=tenant_id,
                    level=chunk.level,
                    seq=chunk.seq,
                    text=chunk.text,
                    token_count=chunk.token_count,
                    embedding=embedding,
                )
                stmt = stmt.on_conflict_do_update(index_elements=["id"], set_={"text": stmt.excluded.text})
                await session.execute(stmt)
        return len(chunks)

    async def semantic_search(self, tenant_id: str, embedding: Sequence[float], *, k: int) -> list[tuple[Chunk, float]]:
        async with self._session_factory() as session, session.begin():
            rows = (
                await session.execute(
                    select(ChunkRow, (1 - ChunkRow.embedding.cosine_distance(embedding)).label("sim"))
                    .where(ChunkRow.tenant_id == tenant_id)
                    .order_by(ChunkRow.embedding.cosine_distance(embedding))
                    .limit(k)
                )
            ).all()
        return [(_chunk(row[0]), float(row[1])) for row in rows]

    async def lexical_search(self, tenant_id: str, text: str, *, k: int) -> list[tuple[Chunk, float]]:
        query = _lexical_tsquery(text)
        rank = safunc.ts_rank_cd(ChunkRow.tsv, query)
        async with self._session_factory() as session, session.begin():
            rows = (
                await session.execute(
                    select(ChunkRow, rank.label("rank"))
                    .where(ChunkRow.tenant_id == tenant_id, ChunkRow.tsv.op("@@")(query))
                    .order_by(rank.desc())
                    .limit(k)
                )
            ).all()
        return [(_chunk(row[0]), float(row[1])) for row in rows]


class PgDocumentRepository(DocumentRepositoryPort):
    def __init__(self, session_factory: SessionFactoryAlias | None = None, settings: AtlasSettings | None = None) -> None:
        self._session_factory = session_factory or create_engine_and_session(
            settings.postgres  # type: ignore[union-attr]
        )[1]

    async def add(self, document: Document, content: bytes) -> None:
        async with self._session_factory() as session, session.begin():
            # An attempt that produced no chunks is replaced, so a document
            # that failed after its row was committed — a guard rejection, a
            # 504 from the embedder — is retryable (BUG-008). A document that
            # did produce chunks still collides on (tenant_id, checksum), which
            # the API surfaces as 409 "already ingested". Chunks and content
            # cascade from the document row.
            await session.execute(
                delete(DocumentRow).where(
                    DocumentRow.tenant_id == document.tenant_id,
                    DocumentRow.checksum == document.checksum,
                    ~exists(select(ChunkRow.id).where(ChunkRow.document_id == DocumentRow.id)),
                )
            )
            session.add(
                DocumentRow(
                    id=document.id,
                    tenant_id=document.tenant_id,
                    source_uri=document.source_uri,
                    mime=document.mime,
                    status=document.status.value,
                    checksum=document.checksum,
                    error=document.error,
                )
            )
            await session.execute(
                pg_insert(DocumentContentRow).values(document_id=document.id, content=content)
            )

    async def get(self, tenant_id: str, document_id: uuid.UUID) -> Document | None:
        async with self._session_factory() as session, session.begin():
            row = (
                await session.execute(
                    select(DocumentRow).where(
                        DocumentRow.tenant_id == tenant_id, DocumentRow.id == document_id
                    )
                )
            ).scalar_one_or_none()
        return _doc(row) if row is not None else None

    async def load_content(self, tenant_id: str, document_id: uuid.UUID) -> bytes | None:
        async with self._session_factory() as session, session.begin():
            row = (
                await session.execute(
                    select(DocumentContentRow)
                    .join(DocumentRow, DocumentRow.id == DocumentContentRow.document_id)
                    .where(
                        DocumentRow.tenant_id == tenant_id,
                        DocumentContentRow.document_id == document_id,
                    )
                )
            ).scalar_one_or_none()
        return row.content if row is not None else None

    async def set_status(
        self, tenant_id: str, document_id: uuid.UUID, status: DocumentStatus, *, error: str | None = None
    ) -> None:
        async with self._session_factory() as session, session.begin():
            await session.execute(
                update(DocumentRow)
                .where(DocumentRow.tenant_id == tenant_id, DocumentRow.id == document_id)
                .values(status=status.value, error=error)
            )

    async def list_latest(self, tenant_id: str, *, limit: int) -> list[Document]:
        async with self._session_factory() as session, session.begin():
            rows = (
                await session.execute(
                    select(DocumentRow)
                    .where(DocumentRow.tenant_id == tenant_id)
                    .order_by(DocumentRow.created_at.desc())
                    .limit(limit)
                )
            ).scalars()
        return [_doc(row) for row in rows]

    async def delete(self, tenant_id: str, document_id: uuid.UUID) -> bool:
        async with self._session_factory() as session, session.begin():
            result = await session.execute(
                delete(DocumentRow).where(DocumentRow.tenant_id == tenant_id, DocumentRow.id == document_id)
            )
        return bool(result.rowcount)  # type: ignore[attr-defined, no-any-return]
