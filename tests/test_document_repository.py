"""Ingest is retryable: re-adding the same bytes replaces the earlier attempt."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from sqlalchemy import Delete, Insert
from sqlalchemy.dialects import postgresql

from atlas_core.domain.models import Document, DocumentStatus
from atlas_vector.repositories import PgDocumentRepository
from atlas_vector.schema import DocumentRow


class _RecordingSession:
    """Stands in for an `AsyncSession` and records statement order."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def __aenter__(self) -> _RecordingSession:
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    def begin(self) -> _RecordingSession:
        return self

    async def execute(self, statement: Any) -> None:
        self.events.append(statement)

    def add(self, instance: Any) -> None:
        self.events.append(instance)


def _document(checksum: str) -> Document:
    return Document(
        id=uuid.uuid4(),
        tenant_id="atlas-test",
        source_uri="docs/page.md",
        mime="text/markdown",
        status=DocumentStatus.PROCESSING,
        checksum=checksum,
    )


def _add(session: _RecordingSession, document: Document) -> None:
    repository = PgDocumentRepository(session_factory=lambda: session)
    asyncio.run(repository.add(document, b"body"))


def test_add_deletes_a_previous_attempt_before_inserting() -> None:
    """BUG-008: a failed document kept its checksum, so it could never be retried."""
    session = _RecordingSession()
    document = _document("abc123")

    _add(session, document)

    delete_stmt, row, insert_stmt = session.events
    assert isinstance(delete_stmt, Delete)
    assert isinstance(row, DocumentRow)
    assert isinstance(insert_stmt, Insert)
    assert row.id == document.id
    assert row.checksum == "abc123"


def test_add_scopes_the_replacement_to_the_tenant_and_checksum() -> None:
    session = _RecordingSession()
    _add(session, _document("abc123"))

    compiled = session.events[0].compile(dialect=postgresql.dialect())
    sql = str(compiled)

    assert "DELETE FROM documents" in sql
    assert "documents.tenant_id" in sql
    assert "documents.checksum" in sql
    assert set(compiled.params.values()) == {"atlas-test", "abc123"}


def test_add_only_replaces_an_attempt_that_produced_no_chunks() -> None:
    """A document that already has chunks must still conflict (HTTP 409)."""
    session = _RecordingSession()
    _add(session, _document("abc123"))

    sql = str(session.events[0].compile(dialect=postgresql.dialect()))

    assert "EXISTS" in sql.upper()
    assert "chunks.document_id = documents.id" in sql
