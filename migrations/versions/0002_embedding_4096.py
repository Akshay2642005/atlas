"""embedding width 1024 -> 4096, drop the ANN index pgvector cannot build

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-26

pgvector caps ANN indexes at 2,000 dimensions for `vector` and 4,000 for
`halfvec`, so a 4096-dim column cannot carry HNSW or IVFFlat. Dense search
therefore runs exactly (`ORDER BY embedding <=> $1`). The GIN index on `tsv`
is unaffected.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Existing 1024-dim vectors cannot be reinterpreted as 4096-dim ones, and
    # the raw source is still in `document_contents`, so the corpus must be
    # re-ingested. Flag it instead of leaving documents silently chunk-less.
    op.execute(
        """
        UPDATE documents
           SET status = 'failed',
               error = 're-ingest required: embedding dimension changed 1024 -> 4096'
         WHERE id IN (SELECT DISTINCT document_id FROM chunks)
        """
    )
    op.execute("DELETE FROM chunks")
    op.drop_index("chunks_hnsw", table_name="chunks")
    op.alter_column(
        "chunks",
        "embedding",
        existing_type=Vector(1024),
        type_=Vector(4096),
        existing_nullable=False,
    )


def downgrade() -> None:
    # Symmetric: 4096-dim vectors cannot become 1024-dim ones.
    op.execute("DELETE FROM chunks")
    op.alter_column(
        "chunks",
        "embedding",
        existing_type=Vector(4096),
        type_=Vector(1024),
        existing_nullable=False,
    )
    op.create_index(
        "chunks_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.execute(
        sa.text(
            """
            UPDATE documents
               SET status = 'failed',
                   error = 're-ingest required: embedding dimension changed 4096 -> 1024'
            """
        )
    )
