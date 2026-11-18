"""retire the pre-versioning content path

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-31

0004 created the version model and the backfill populated it. This migration
removes what the model replaced, and only after `--verify` passed on every local
database:

- `chunks.document_version_id` becomes NOT NULL. Every chunk belongs to exactly
  one version; retrieval joins through it to the head.
- `document_contents` is dropped. The bytes live in the object store at the key
  recorded on `blobs.storage_key`.
- `documents.checksum`, `.mime`, `.status`, `.error` are dropped. They are now
  derived from the newest version, so they cannot disagree with the lineage.
- `uq_documents_tenant_checksum` gives way to `UNIQUE (tenant_id, source_uri)`:
  a document is an identity plus a lineage, and the old constraint made an edit
  impossible — it could only collide or duplicate.

Downgrade restores the columns and the old constraint but cannot restore
`document_contents`: the bytes are in object storage now, and re-importing them
is the backfill in reverse. That asymmetry is deliberate and recorded here.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("uq_documents_tenant_checksum", "documents", type_="unique")

    op.alter_column(
        "chunks",
        "document_version_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=False,
    )

    op.drop_table("document_contents")

    op.drop_column("documents", "error")
    op.drop_column("documents", "checksum")
    op.drop_column("documents", "status")
    op.drop_column("documents", "mime")

    op.create_unique_constraint(
        "uq_documents_tenant_source", "documents", ["tenant_id", "source_uri"]
    )


def downgrade() -> None:
    op.drop_constraint("uq_documents_tenant_source", "documents", type_="unique")

    op.add_column(
        "documents", sa.Column("mime", sa.String(length=64), nullable=False, server_default="")
    )
    op.add_column(
        "documents",
        sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
    )
    op.add_column(
        "documents", sa.Column("checksum", sa.String(length=64), nullable=False, server_default="")
    )
    op.add_column("documents", sa.Column("error", sa.Text(), nullable=True))
    # Restore what the lineage knows. The bytes themselves are in object
    # storage; a downgrade cannot copy them back, and `document_contents` comes
    # back empty on purpose rather than lying about its contents.
    op.execute(
        """
        UPDATE documents d SET
            mime = v.mime,
            status = v.status,
            checksum = v.checksum,
            error = v.error
        FROM document_versions v
        WHERE v.document_id = d.id
          AND v.seq = (SELECT max(seq) FROM document_versions WHERE document_id = d.id)
        """
    )
    op.alter_column("documents", "mime", server_default=None)
    op.alter_column("documents", "status", server_default=None)

    op.create_table(
        "document_contents",
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("content", sa.LargeBinary(), nullable=False),
    )

    op.alter_column(
        "chunks",
        "document_version_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=True,
    )

    op.create_unique_constraint(
        "uq_documents_tenant_checksum", "documents", ["tenant_id", "checksum"]
    )
