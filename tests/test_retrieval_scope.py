"""Both retrieval branches must be scoped to the head version — identically.

BUG-006 was two queries drifting apart: the dense branch and the lexical branch
were written separately and stopped agreeing on what "searchable" meant. The fix
was a shared predicate. This test is what keeps it shared, by compiling the two
statements and asserting that each carries the same head subquery.
"""

from __future__ import annotations

from sqlalchemy.dialects import postgresql

from atlas_vector.repositories import lexical_statement, semantic_statement


def _sql(statement: object) -> str:
    return str(statement.compile(dialect=postgresql.dialect()))  # type: ignore[attr-defined]


def _head_scope(sql: str) -> bool:
    # The predicate is a correlated subquery on the document's head pointer.
    return (
        "chunks.document_version_id = (SELECT documents.head_version_id" in sql.replace("\n", " ")
    )


def test_dense_branch_is_scoped_to_the_head_version() -> None:
    sql = _sql(semantic_statement("tenant-a", [0.1, 0.2], k=10))

    assert _head_scope(sql)
    assert "chunks.tenant_id = " in sql


def test_lexical_branch_is_scoped_to_the_head_version() -> None:
    sql = _sql(lexical_statement("tenant-a", "pgvector halfvec", k=10))

    assert _head_scope(sql)
    assert "chunks.tenant_id = " in sql


def test_the_two_branches_share_the_same_scope_predicate() -> None:
    dense = _sql(semantic_statement("tenant-a", [0.1], k=5))
    lexical = _sql(lexical_statement("tenant-a", "hello world", k=5))

    scope = "chunks.document_version_id = (SELECT documents.head_version_id"
    assert scope in dense.replace("\n", " ")
    assert scope in lexical.replace("\n", " ")
