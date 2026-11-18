"""SQLAlchemy schema mirroring Alembic migrations."""

from __future__ import annotations

import datetime
import uuid
from typing import Any

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import Computed, DateTime, ForeignKey, Index, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.dialects.postgresql import UUID as Uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class BlobRow(Base):
    """Immutable bytes, content-addressed and scoped to one tenant.

    `sha256` alone would dedupe across tenants, which leaks existence: a tenant
    could confirm another tenant holds a document by uploading the same bytes
    and observing "unchanged". The unique key is therefore `(tenant_id, sha256)`.
    """

    __tablename__ = "blobs"
    __table_args__ = (UniqueConstraint("tenant_id", "sha256", name="uq_blobs_tenant_sha256"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(nullable=False)
    storage_key: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class DocumentRow(Base):
    """Document identity. Everything about the bytes lives on its versions."""

    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("tenant_id", "source_uri", name="uq_documents_tenant_source"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_uri: Mapped[str] = mapped_column(Text, nullable=False)
    # The only mutable pointer in the version model. A version flips it only
    # once it is fully indexed, so a half-ingested version is never searchable.
    # `use_alter` breaks the documents <-> document_versions cycle in DDL.
    head_version_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid,
        ForeignKey(
            "document_versions.id",
            ondelete="SET NULL",
            name="fk_documents_head_version",
            use_alter=True,
        ),
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class DocumentVersionRow(Base):
    """One immutable entry in a document's lineage.

    History is append-only: an edit inserts a new row, and a revert inserts a
    new row pointing at an old blob. Nothing is ever rewritten, so a citation
    that pins a chunk of an old version cannot dangle.
    """

    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint("document_id", "seq", name="uq_document_versions_document_seq"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    seq: Mapped[int] = mapped_column(nullable=False)
    blob_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("blobs.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    mime: Mapped[str] = mapped_column(String(64), nullable=False)
    message: Mapped[str | None] = mapped_column(Text)
    parent_version_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("document_versions.id", ondelete="SET NULL")
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    error: Mapped[str | None] = mapped_column(Text)
    chunk_count: Mapped[int] = mapped_column(nullable=False, default=0, server_default="0")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ChunkRow(Base):
    __tablename__ = "chunks"
    # No HNSW/IVFFlat index: pgvector caps `halfvec` at 4,000 dimensions for
    # ANN indexes, so a 4096-dim column can only be searched exactly. The
    # column is `halfvec` rather than `vector` because it halves storage and
    # exact-scan I/O at the same precision that matters for cosine ranking.
    # Revisit if the embedding width drops to <= 4,000.
    __table_args__ = (Index("chunks_tsv_gin", "tsv", postgresql_using="gin"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Which version of the document this chunk belongs to. Every chunk belongs
    # to exactly one version, and retrieval only sees the head version's chunks.
    document_version_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("document_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    level: Mapped[int] = mapped_column(nullable=False, default=0, server_default="0")
    seq: Mapped[int] = mapped_column(nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(nullable=False)
    embedding: Mapped[Any] = mapped_column(HALFVEC(4096), nullable=False)
    tsv: Mapped[Any] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True), nullable=False
    )


