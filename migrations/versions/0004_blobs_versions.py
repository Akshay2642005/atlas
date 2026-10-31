"""blobs and immutable document versions

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-31

Purely additive, on purpose. This migration is the risky half of the versioning
epic — it runs against live corpora — so it changes structure and nothing else:

- `blobs` holds content-addressed bytes, unique per `(tenant_id, sha256)`.
  Global uniqueness on `sha256` would dedupe across tenants and leak existence.
- `document_versions` is the append-only lineage; a revert later inserts a new
  row rather than rewriting one.
- `documents.head_version_id` is the single mutable pointer. It is nullable and
  the FK is `use_alter` because `documents` and `document_versions` reference
  each other.
- `chunks.document_version_id` is nullable: it is filled by the backfill that
  runs after this migration, and made NOT NULL once the pipeline writes
  versions itself.

Deliberately NOT dropped here: `uq_documents_tenant_checksum`. The API's 409
for duplicate content is that constraint raising IntegrityError, and it must
survive until the commit-semantics rewrite replaces it. `document_contents`
also stays until every row has been copied to the store and verified.

Indexes: a btree on `(document_id, seq)` also serves "newest first" because
Postgres scans it backwards, so no separate DESC index is created.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "blobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("storage_key", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("tenant_id", "sha256", name="uq_blobs_tenant_sha256"),
    )
    op.create_index("ix_blobs_tenant_id", "blobs", ["tenant_id"])

    op.create_table(
        "document_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column(
            "blob_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("blobs.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("checksum", sa.String(length=64), nullable=False),
        sa.Column("mime", sa.String(length=64), nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column(
            "parent_version_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("document_versions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("document_id", "seq", name="uq_document_versions_document_seq"),
    )
    op.create_index("ix_document_versions_document_id", "document_versions", ["document_id"])
    op.create_index("ix_document_versions_blob_id", "document_versions", ["blob_id"])

    op.add_column("documents", sa.Column("head_version_id", postgresql.UUID(as_uuid=True)))
    op.create_foreign_key(
        "fk_documents_head_version",
        "documents",
        "document_versions",
        ["head_version_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.add_column("chunks", sa.Column("document_version_id", postgresql.UUID(as_uuid=True)))
    op.create_foreign_key(
        "fk_chunks_document_version",
        "chunks",
        "document_versions",
        ["document_version_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index("ix_chunks_document_version_id", "chunks", ["document_version_id"])


def downgrade() -> None:
    op.drop_index("ix_chunks_document_version_id", table_name="chunks")
    op.drop_constraint("fk_chunks_document_version", "chunks", type_="foreignkey")
    op.drop_column("chunks", "document_version_id")

    op.drop_constraint("fk_documents_head_version", "documents", type_="foreignkey")
    op.drop_column("documents", "head_version_id")

    op.drop_index("ix_document_versions_blob_id", table_name="document_versions")
    op.drop_index("ix_document_versions_document_id", table_name="document_versions")
    op.drop_table("document_versions")

    op.drop_index("ix_blobs_tenant_id", table_name="blobs")
    op.drop_table("blobs")
