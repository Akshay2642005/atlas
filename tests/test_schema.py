"""Schema guards: the embedding column must match the configured width."""

from __future__ import annotations

from sqlalchemy.dialects import postgresql

from atlas_vector.schema import ChunkRow


def test_embedding_column_is_halfvec_4096() -> None:
    column = ChunkRow.__table__.c.embedding
    dialect = postgresql.dialect()

    assert column.type.compile(dialect=dialect) == "HALFVEC(4096)"
    assert column.nullable is False


def test_cosine_distance_uses_the_pgvector_operator() -> None:
    expression = ChunkRow.embedding.cosine_distance([0.0] * 4096)

    assert "<=>" in str(expression.compile(dialect=postgresql.dialect()))
