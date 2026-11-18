"""pgVector repositories implementing core ports. Tenant isolation on every query."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import ColumnElement, and_, delete, select, update
from sqlalchemy import func as safunc
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from atlas_core.config import AtlasSettings
from atlas_core.domain.models import (
    Blob,
    Chunk,
    CommitResult,
    Document,
    DocumentStatus,
    DocumentVersion,
)
from atlas_core.domain.ports import DocumentRepositoryPort, VectorRepositoryPort
from atlas_vector.engine import SessionFactory, create_engine_and_session
from atlas_vector.schema import BlobRow, ChunkRow, DocumentRow, DocumentVersionRow

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


def searchable() -> ColumnElement[bool]:
    """The one predicate both retrieval branches filter by.

    A chunk is searchable only if it belongs to its document's head version.
    Because the head pointer moves *after* a version reaches `indexed`, an
    unfinished or failed version is in the lineage and invisible to search.

    Dense and lexical share this function rather than each writing the join:
    BUG-006 was exactly two branches drifting apart, and a copy would invite it
    again.
    """
    head = (
        select(DocumentRow.head_version_id)
        .where(DocumentRow.id == ChunkRow.document_id)
        .scalar_subquery()
    )
    return ChunkRow.document_version_id == head


def semantic_statement(
    tenant_id: str, embedding: Sequence[float], *, k: int
) -> Any:
    """Dense branch. Exposed so a test can prove it shares the scope predicate."""
    return (
        select(ChunkRow, (1 - ChunkRow.embedding.cosine_distance(embedding)).label("sim"))
        .where(ChunkRow.tenant_id == tenant_id, searchable())
        .order_by(ChunkRow.embedding.cosine_distance(embedding))
        .limit(k)
    )


def lexical_statement(tenant_id: str, text: str, *, k: int) -> Any:
    """Lexical branch, with the same `searchable()` predicate as the dense one."""
    query = _lexical_tsquery(text)
    rank = safunc.ts_rank_cd(ChunkRow.tsv, query)
    return (
        select(ChunkRow, rank.label("rank"))
        .where(
            ChunkRow.tenant_id == tenant_id,
            ChunkRow.tsv.op("@@")(query),
            searchable(),
        )
        .order_by(rank.desc())
        .limit(k)
    )


def _chunk(row: ChunkRow) -> Chunk:
    return Chunk(
        id=row.id,
        document_id=row.document_id,
        tenant_id=row.tenant_id,
        seq=row.seq,
        level=row.level,
        text=row.text,
        token_count=row.token_count,
        document_version_id=row.document_version_id,
    )


def _version(row: DocumentVersionRow, *, storage_key: str = "", size_bytes: int = 0) -> DocumentVersion:
    return DocumentVersion(
        id=row.id,
        document_id=row.document_id,
        seq=row.seq,
        checksum=row.checksum,
        mime=row.mime,
        status=DocumentStatus(row.status),
        chunk_count=row.chunk_count,
        parent_version_id=row.parent_version_id,
        message=row.message,
        error=row.error,
        created_at=row.created_at,
        storage_key=storage_key,
        size_bytes=size_bytes,
    )


def _document(row: DocumentRow, version: DocumentVersion | None) -> Document:
    """Derive the document view from its lineage.

    Status, mime and checksum are read from the newest version rather than
    duplicated on `documents`, so the API cannot report a state the lineage
    does not hold. `head_version_id` is what retrieval is actually serving; the
    newest version may be a failed attempt on top of it.
    """
    return Document(
        id=row.id,
        tenant_id=row.tenant_id,
        source_uri=row.source_uri,
        mime=version.mime if version is not None else "",
        status=version.status if version is not None else DocumentStatus.PENDING,
        checksum=version.checksum if version is not None else "",
        created_at=row.created_at,
        error=version.error if version is not None else None,
        head_version_id=row.head_version_id,
    )


def _latest_version() -> Any:
    """Join every document to its newest version (highest seq)."""
    newest_seq = (
        select(safunc.max(DocumentVersionRow.seq))
        .where(DocumentVersionRow.document_id == DocumentRow.id)
        .scalar_subquery()
    )
    return and_(
        DocumentVersionRow.document_id == DocumentRow.id,
        DocumentVersionRow.seq == newest_seq,
    )


class PgVectorRepository(VectorRepositoryPort):
    def __init__(
        self, session_factory: SessionFactoryAlias | None = None, settings: AtlasSettings | None = None
    ) -> None:
        self._session_factory = session_factory or create_engine_and_session(
            settings.postgres  # type: ignore[union-attr]
        )[1]

    async def upsert_chunks(
        self,
        tenant_id: str,
        document_version_id: uuid.UUID,
        chunks: Sequence[tuple[Chunk, Sequence[float]]],
    ) -> int:
        async with self._session_factory() as session, session.begin():
            for chunk, embedding in chunks:
                stmt = pg_insert(ChunkRow).values(
                    id=chunk.id,
                    document_id=chunk.document_id,
                    document_version_id=document_version_id,
                    tenant_id=tenant_id,
                    level=chunk.level,
                    seq=chunk.seq,
                    text=chunk.text,
                    token_count=chunk.token_count,
                    embedding=embedding,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=["id"],
                    set_={"text": stmt.excluded.text, "document_version_id": document_version_id},
                )
                await session.execute(stmt)
        return len(chunks)

    async def semantic_search(
        self, tenant_id: str, embedding: Sequence[float], *, k: int
    ) -> list[tuple[Chunk, float]]:
        async with self._session_factory() as session, session.begin():
            rows = (await session.execute(semantic_statement(tenant_id, embedding, k=k))).all()
        return [(_chunk(row[0]), float(row[1])) for row in rows]

    async def lexical_search(
        self, tenant_id: str, text: str, *, k: int
    ) -> list[tuple[Chunk, float]]:
        async with self._session_factory() as session, session.begin():
            rows = (await session.execute(lexical_statement(tenant_id, text, k=k))).all()
        return [(_chunk(row[0]), float(row[1])) for row in rows]


async def _upsert_blob(session: AsyncSession, tenant_id: str, blob: Blob) -> uuid.UUID:
    await session.execute(
        pg_insert(BlobRow)
        .values(
            id=uuid.uuid4(),
            tenant_id=tenant_id,
            sha256=blob.sha256,
            size_bytes=blob.size_bytes,
            storage_key=blob.storage_key,
        )
        .on_conflict_do_nothing(index_elements=["tenant_id", "sha256"])
    )
    blob_id = (
        await session.execute(
            select(BlobRow.id).where(
                BlobRow.tenant_id == tenant_id, BlobRow.sha256 == blob.sha256
            )
        )
    ).scalar_one()
    return uuid.UUID(str(blob_id))


class PgDocumentRepository(DocumentRepositoryPort):
    def __init__(
        self, session_factory: SessionFactoryAlias | None = None, settings: AtlasSettings | None = None
    ) -> None:
        self._session_factory = session_factory or create_engine_and_session(
            settings.postgres  # type: ignore[union-attr]
        )[1]

    async def commit(
        self,
        *,
        tenant_id: str,
        source_uri: str,
        blob: Blob,
        mime: str,
        message: str | None = None,
    ) -> CommitResult:
        async with self._session_factory() as session, session.begin():
            # Lock the identity row so two concurrent uploads of the same URI
            # allocate seq numbers in order instead of colliding on (id, seq).
            row = (
                await session.execute(
                    select(DocumentRow)
                    .where(
                        DocumentRow.tenant_id == tenant_id,
                        DocumentRow.source_uri == source_uri,
                    )
                    .with_for_update()
                )
            ).scalar_one_or_none()

            if row is None:
                row = DocumentRow(
                    id=uuid.uuid4(),
                    tenant_id=tenant_id,
                    source_uri=source_uri,
                    head_version_id=None,
                )
                session.add(row)
                await session.flush()
                version = await self._append_version(
                    session, row, blob, mime, seq=1, message=message, parent_version_id=None
                )
                return CommitResult(
                    document=_document(row, version), version=version, created=True
                )

            latest = (
                await session.execute(
                    select(DocumentVersionRow)
                    .where(DocumentVersionRow.document_id == row.id)
                    .order_by(DocumentVersionRow.seq.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()

            if latest is not None and latest.checksum == blob.sha256:
                if latest.status == DocumentStatus.INDEXED.value:
                    # Same bytes as what is already served: nothing to do.
                    served = _version(
                        latest, storage_key=blob.storage_key, size_bytes=blob.size_bytes
                    )
                    return CommitResult(
                        document=_document(row, served), version=served, created=False
                    )
                # Same bytes, but that attempt never finished. Retry it in place
                # rather than appending seq+1 for identical content.
                await session.execute(
                    update(DocumentVersionRow)
                    .where(DocumentVersionRow.id == latest.id)
                    .values(status=DocumentStatus.PENDING.value, error=None)
                )
                latest.status = DocumentStatus.PENDING.value
                latest.error = None
                retried = _version(
                    latest, storage_key=blob.storage_key, size_bytes=blob.size_bytes
                )
                return CommitResult(
                    document=_document(row, retried), version=retried, created=False
                )

            version = await self._append_version(
                session,
                row,
                blob,
                mime,
                seq=(latest.seq + 1) if latest is not None else 1,
                message=message,
                parent_version_id=row.head_version_id,
            )
            newest = latest if latest is not None else None
            return CommitResult(
                document=_document(row, _version(newest) if newest is not None else version),
                version=version,
                created=True,
            )

    async def _append_version(
        self,
        session: AsyncSession,
        row: DocumentRow,
        blob: Blob,
        mime: str,
        *,
        seq: int,
        message: str | None,
        parent_version_id: uuid.UUID | None,
    ) -> DocumentVersion:
        blob_id = await _upsert_blob(session, row.tenant_id, blob)
        version_row = DocumentVersionRow(
            id=uuid.uuid4(),
            document_id=row.id,
            seq=seq,
            blob_id=blob_id,
            checksum=blob.sha256,
            mime=mime,
            message=message,
            parent_version_id=parent_version_id,
            status=DocumentStatus.PENDING.value,
            chunk_count=0,
        )
        session.add(version_row)
        await session.flush()
        return _version(
            version_row, storage_key=blob.storage_key, size_bytes=blob.size_bytes
        )

    async def get(self, tenant_id: str, document_id: uuid.UUID) -> Document | None:
        async with self._session_factory() as session, session.begin():
            row = (
                await session.execute(
                    select(DocumentRow, DocumentVersionRow)
                    .outerjoin(DocumentVersionRow, _latest_version())
                    .where(DocumentRow.tenant_id == tenant_id, DocumentRow.id == document_id)
                )
            ).first()
        if row is None:
            return None
        return _document(row[0], _version(row[1]) if row[1] is not None else None)

    async def get_version(
        self, tenant_id: str, version_id: uuid.UUID
    ) -> DocumentVersion | None:
        async with self._session_factory() as session, session.begin():
            row = (
                await session.execute(
                    select(DocumentVersionRow, BlobRow)
                    .join(DocumentRow, DocumentRow.id == DocumentVersionRow.document_id)
                    .join(BlobRow, BlobRow.id == DocumentVersionRow.blob_id)
                    .where(DocumentRow.tenant_id == tenant_id, DocumentVersionRow.id == version_id)
                )
            ).first()
        if row is None:
            return None
        return _version(row[0], storage_key=row[1].storage_key, size_bytes=row[1].size_bytes)

    async def list_versions(
        self, tenant_id: str, document_id: uuid.UUID, *, limit: int
    ) -> list[DocumentVersion]:
        async with self._session_factory() as session, session.begin():
            rows = (
                await session.execute(
                    select(DocumentVersionRow, BlobRow)
                    .join(DocumentRow, DocumentRow.id == DocumentVersionRow.document_id)
                    .join(BlobRow, BlobRow.id == DocumentVersionRow.blob_id)
                    .where(
                        DocumentRow.tenant_id == tenant_id,
                        DocumentVersionRow.document_id == document_id,
                    )
                    .order_by(DocumentVersionRow.seq.desc())
                    .limit(limit)
                )
            ).all()
        return [
            _version(row[0], storage_key=row[1].storage_key, size_bytes=row[1].size_bytes)
            for row in rows
        ]

    async def set_version_status(
        self,
        tenant_id: str,
        version_id: uuid.UUID,
        status: DocumentStatus,
        *,
        error: str | None = None,
        chunk_count: int | None = None,
    ) -> None:
        values: dict[str, Any] = {"status": status.value, "error": error}
        if chunk_count is not None:
            values["chunk_count"] = chunk_count
        async with self._session_factory() as session, session.begin():
            await session.execute(
                update(DocumentVersionRow)
                .where(
                    DocumentVersionRow.id == version_id,
                    DocumentVersionRow.document_id.in_(
                        select(DocumentRow.id).where(DocumentRow.tenant_id == tenant_id)
                    ),
                )
                .values(**values)
            )

    async def set_head(
        self, tenant_id: str, document_id: uuid.UUID, version_id: uuid.UUID
    ) -> None:
        async with self._session_factory() as session, session.begin():
            await session.execute(
                update(DocumentRow)
                .where(DocumentRow.tenant_id == tenant_id, DocumentRow.id == document_id)
                .values(head_version_id=version_id)
            )

    async def resolve_serving_version(self, document: Document) -> DocumentVersion | None:
        if document.head_version_id is None:
            return None
        return await self.get_version(document.tenant_id, document.head_version_id)

    async def list_latest(self, tenant_id: str, *, limit: int) -> list[Document]:
        async with self._session_factory() as session, session.begin():
            rows = (
                await session.execute(
                    select(DocumentRow, DocumentVersionRow)
                    .outerjoin(DocumentVersionRow, _latest_version())
                    .where(DocumentRow.tenant_id == tenant_id)
                    .order_by(DocumentRow.created_at.desc())
                    .limit(limit)
                )
            ).all()
        return [
            _document(row[0], _version(row[1]) if row[1] is not None else None) for row in rows
        ]

    async def delete(self, tenant_id: str, document_id: uuid.UUID) -> bool:
        async with self._session_factory() as session, session.begin():
            result = await session.execute(
                delete(DocumentRow).where(
                    DocumentRow.tenant_id == tenant_id, DocumentRow.id == document_id
                )
            )
        return bool(result.rowcount)  # type: ignore[attr-defined, no-any-return]


__all__ = [
    "PgDocumentRepository",
    "PgVectorRepository",
    "lexical_statement",
    "searchable",
    "semantic_statement",
]
