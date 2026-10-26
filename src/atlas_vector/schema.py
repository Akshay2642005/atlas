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


class DocumentRow(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("tenant_id", "checksum", name="uq_documents_tenant_checksum"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_uri: Mapped[str] = mapped_column(Text, nullable=False)
    mime: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
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
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    level: Mapped[int] = mapped_column(nullable=False, default=0, server_default="0")
    seq: Mapped[int] = mapped_column(nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(nullable=False)
    embedding: Mapped[Any] = mapped_column(HALFVEC(4096), nullable=False)
    tsv: Mapped[Any] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True), nullable=False
    )


class DocumentContentRow(Base):
    """Raw source payload (BYTEA) kept out of `documents` for clean lifecycle mgmt."""

    __tablename__ = "document_contents"

    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    content: Mapped[bytes] = mapped_column(nullable=False)
