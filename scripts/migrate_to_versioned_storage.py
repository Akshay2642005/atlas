"""Move document bytes out of Postgres into content-addressed blobs.

One pass per document:

    document_contents.content
      -> ContentStorePort at {tenant}/{sha256[:2]}/{sha256}
      -> a `blobs` row
      -> `document_versions` seq 1, carrying the document's current status
      -> `documents.head_version_id`
      -> every `chunks.document_version_id`

Idempotent by construction: the content key is derived from the bytes, the blob
unique key is `(tenant_id, sha256)`, and the version is `(document_id, seq)`. A
re-run skips documents that already have a head, so a crash half way through is
resumed by running the same command again.

    uv run python scripts/migrate_to_versioned_storage.py --dry-run
    uv run python scripts/migrate_to_versioned_storage.py
    uv run python scripts/migrate_to_versioned_storage.py --verify

`--verify` rehashes every head blob rather than trusting the insert. It is the
gate that decides whether `document_contents` can ever be dropped.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from atlas_core.config import AtlasSettings, load_settings
from atlas_core.domain.ports import ContentStorePort
from atlas_storage.keys import content_key
from atlas_storage.obstore_store import build_content_store
from atlas_vector.engine import create_engine_and_session

BATCH = 64


@dataclass(frozen=True, slots=True)
class VersionPlan:
    """Everything one document needs, decided before anything is written."""

    sha256: str
    size_bytes: int
    storage_key: str
    checksum: str
    mime: str
    status: str
    chunk_count: int
    seq: int = 1


@dataclass
class Counters:
    documents: int = 0
    skipped: int = 0
    written: int = 0
    chunks_linked: int = 0
    problems: list[str] | None = None

    def problem(self, message: str) -> None:
        if self.problems is None:
            self.problems = []
        self.problems.append(message)


def plan_version(
    *,
    tenant_id: str,
    checksum: str,
    mime: str,
    status: str,
    chunk_count: int,
    content: bytes,
) -> VersionPlan:
    """Derive the blob and version for one document's bytes.

    The stored checksum is recomputed rather than trusted: if `documents.checksum`
    and the bytes disagree, the row was written by something other than the
    pipeline and the content key would be a lie.
    """
    digest = hashlib.sha256(content).hexdigest()
    if digest != checksum:
        raise ValueError(
            f"checksum mismatch for tenant {tenant_id!r}: stored {checksum}, bytes {digest}"
        )
    return VersionPlan(
        sha256=digest,
        size_bytes=len(content),
        storage_key=content_key(tenant_id, digest),
        checksum=checksum,
        mime=mime,
        status=status,
        chunk_count=chunk_count,
    )


async def _fetch_batch(
    conn: AsyncConnection, after: uuid.UUID | None, tenant: str | None
) -> list[Any]:
    rows = await conn.execute(
        text(
            """
            SELECT d.id,
                   d.tenant_id,
                   d.checksum,
                   d.mime,
                   d.status,
                   d.head_version_id,
                   (SELECT count(*) FROM chunks c WHERE c.document_id = d.id) AS chunk_count,
                   dc.content
            FROM documents d
            LEFT JOIN document_contents dc ON dc.document_id = d.id
            WHERE (CAST(:tenant AS text) IS NULL OR d.tenant_id = :tenant)
              AND (CAST(:after AS uuid) IS NULL OR d.id > CAST(:after AS uuid))
            ORDER BY d.id
            LIMIT :limit
            """
        ),
        {"tenant": tenant, "after": after, "limit": BATCH},
    )
    return list(rows)


async def _upsert_blob(conn: AsyncConnection, row: Any, plan: VersionPlan) -> uuid.UUID:
    await conn.execute(
        text(
            """
            INSERT INTO blobs (id, tenant_id, sha256, size_bytes, storage_key)
            VALUES (:id, :tenant_id, :sha256, :size_bytes, :storage_key)
            ON CONFLICT (tenant_id, sha256) DO NOTHING
            """
        ),
        {
            "id": uuid.uuid4(),
            "tenant_id": row.tenant_id,
            "sha256": plan.sha256,
            "size_bytes": plan.size_bytes,
            "storage_key": plan.storage_key,
        },
    )
    blob_id = (
        await conn.execute(
            text("SELECT id FROM blobs WHERE tenant_id = :tenant_id AND sha256 = :sha256"),
            {"tenant_id": row.tenant_id, "sha256": plan.sha256},
        )
    ).scalar_one()
    return uuid.UUID(str(blob_id))


async def _ensure_version(conn: AsyncConnection, row: Any, plan: VersionPlan) -> uuid.UUID:
    blob_id = await _upsert_blob(conn, row, plan)
    await conn.execute(
        text(
            """
            INSERT INTO document_versions
                (id, document_id, seq, blob_id, checksum, mime, message, status, chunk_count)
            VALUES
                (:id, :document_id, :seq, :blob_id, :checksum, :mime, NULL, :status, :chunk_count)
            ON CONFLICT (document_id, seq) DO NOTHING
            """
        ),
        {
            "id": uuid.uuid4(),
            "document_id": row.id,
            "seq": plan.seq,
            "blob_id": blob_id,
            "checksum": plan.checksum,
            "mime": plan.mime,
            "status": plan.status,
            "chunk_count": plan.chunk_count,
        },
    )
    version_id = (
        await conn.execute(
            text("SELECT id FROM document_versions WHERE document_id = :document_id AND seq = :seq"),
            {"document_id": row.id, "seq": plan.seq},
        )
    ).scalar_one()
    return uuid.UUID(str(version_id))


async def _backfill_document(
    conn: AsyncConnection,
    store: ContentStorePort,
    row: Any,
    counters: Counters,
    *,
    dry_run: bool,
) -> None:
    counters.documents += 1
    if row.content is None:
        counters.problem(f"document {row.id} has no stored content; skipped")
        return

    try:
        plan = plan_version(
            tenant_id=row.tenant_id,
            checksum=row.checksum,
            mime=row.mime,
            status=row.status,
            chunk_count=int(row.chunk_count),
            content=bytes(row.content),
        )
    except ValueError as exc:
        counters.problem(str(exc))
        return

    if dry_run:
        counters.written += 1
        return

    # Object first, then metadata. A crash in between leaves an orphan blob,
    # which the GC job reclaims; the reverse order would leave a row pointing
    # at bytes that do not exist, which no job can repair.
    if await store.head(plan.storage_key) is None:
        await store.put_bytes(plan.storage_key, bytes(row.content))

    version_id = await _ensure_version(conn, row, plan)
    await conn.execute(
        text("UPDATE documents SET head_version_id = :version_id WHERE id = :id"),
        {"version_id": version_id, "id": row.id},
    )
    linked = await conn.execute(
        text(
            """
            UPDATE chunks SET document_version_id = :version_id
            WHERE document_id = :id AND document_version_id IS NULL
            """
        ),
        {"version_id": version_id, "id": row.id},
    )
    counters.written += 1
    counters.chunks_linked += int(linked.rowcount or 0)  # type: ignore[attr-defined]


async def run_backfill(
    engine: AsyncEngine,
    store: ContentStorePort,
    *,
    tenant: str | None,
    dry_run: bool,
) -> Counters:
    counters = Counters()
    after: uuid.UUID | None = None
    while True:
        async with engine.connect() as conn:
            rows = await _fetch_batch(conn, after, tenant)
        if not rows:
            break
        for row in rows:
            if row.head_version_id is not None:
                counters.skipped += 1
                continue
            async with engine.begin() as conn:
                await _backfill_document(conn, store, row, counters, dry_run=dry_run)
        after = rows[-1].id
    return counters


async def verify(
    engine: AsyncEngine, store: ContentStorePort, *, tenant: str | None
) -> Counters:
    """Prove the backfill is complete and the bytes behind it are real."""
    counters = Counters()
    params = {"tenant": tenant}

    async with engine.begin() as conn:
        for row in (
            await conn.execute(
                text(
                    """
                    SELECT id FROM documents
                    WHERE (CAST(:tenant AS text) IS NULL OR tenant_id = :tenant)
                      AND head_version_id IS NULL
                    """
                ),
                params,
            )
        ).all():
            counters.problem(f"document {row.id} has no head version")

        null_chunks = (
            await conn.execute(
                text(
                    """
                    SELECT count(*) FROM chunks
                    WHERE (CAST(:tenant AS text) IS NULL OR tenant_id = :tenant)
                      AND document_version_id IS NULL
                    """
                ),
                params,
            )
        ).scalar_one()
        if null_chunks:
            counters.problem(f"{null_chunks} chunks have no document_version_id")

        mismatched = (
            await conn.execute(
                text(
                    """
                    SELECT count(*)
                    FROM chunks c
                    JOIN document_versions v ON v.id = c.document_version_id
                    WHERE (CAST(:tenant AS text) IS NULL OR c.tenant_id = :tenant)
                      AND v.document_id <> c.document_id
                    """
                ),
                params,
            )
        ).scalar_one()
        if mismatched:
            counters.problem(f"{mismatched} chunks point at a version of another document")

        total_chunks = (
            await conn.execute(
                text(
                    """
                    SELECT count(*) FROM chunks
                    WHERE (CAST(:tenant AS text) IS NULL OR tenant_id = :tenant)
                    """
                ),
                params,
            )
        ).scalar_one()
        declared = (
            await conn.execute(
                text(
                    """
                    SELECT coalesce(sum(v.chunk_count), 0)
                    FROM document_versions v
                    JOIN documents d ON d.id = v.document_id
                    WHERE (CAST(:tenant AS text) IS NULL OR d.tenant_id = :tenant)
                    """
                ),
                params,
            )
        ).scalar_one()
        if int(total_chunks) != int(declared):
            counters.problem(
                f"chunk count drifted: {total_chunks} rows, {declared} declared by versions"
            )
        counters.chunks_linked = int(total_chunks)

        heads = (
            await conn.execute(
                text(
                    """
                    SELECT d.id AS document_id, d.tenant_id, d.checksum,
                           v.id AS version_id, b.sha256, b.size_bytes, b.storage_key
                    FROM documents d
                    JOIN document_versions v ON v.id = d.head_version_id
                    JOIN blobs b ON b.id = v.blob_id
                    WHERE (CAST(:tenant AS text) IS NULL OR d.tenant_id = :tenant)
                    ORDER BY d.id
                    """
                ),
                params,
            )
        ).all()

    for head in heads:
        counters.documents += 1
        if head.sha256 != head.checksum:
            counters.problem(f"document {head.document_id}: version checksum != document checksum")
        meta = await store.head(head.storage_key)
        if meta is None:
            counters.problem(f"document {head.document_id}: blob missing at {head.storage_key}")
            continue
        if meta.size_bytes != head.size_bytes:
            counters.problem(
                f"document {head.document_id}: stored {meta.size_bytes} bytes, "
                f"blob row says {head.size_bytes}"
            )
            continue
        digest = hashlib.sha256()
        stream = await store.open_stream(head.storage_key)
        async for chunk in stream:
            digest.update(chunk)
        if digest.hexdigest() != head.sha256:
            counters.problem(f"document {head.document_id}: blob does not hash to its key")
    return counters


def _report(action: str, counters: Counters) -> int:
    problems = counters.problems or []
    print(
        f"{action}: documents={counters.documents} written={counters.written} "
        f"skipped={counters.skipped} chunks_linked={counters.chunks_linked} "
        f"problems={len(problems)}"
    )
    for problem in problems:
        print(f"  ! {problem}", file=sys.stderr)
    return 1 if problems else 0


async def _main(args: argparse.Namespace) -> int:
    settings: AtlasSettings = load_settings()
    store = build_content_store(settings.storage)
    engine, _ = create_engine_and_session(settings.postgres)
    try:
        if args.verify:
            return _report("verify", await verify(engine, store, tenant=args.tenant))
        counters = await run_backfill(
            engine, store, tenant=args.tenant, dry_run=args.dry_run
        )
        return _report("dry-run" if args.dry_run else "backfill", counters)
    finally:
        await engine.dispose()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", default=None, help="limit to one tenant")
    parser.add_argument(
        "--dry-run", action="store_true", help="decide what would be written, write nothing"
    )
    parser.add_argument(
        "--verify", action="store_true", help="assert the backfill is complete and the bytes are real"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main(_parse_args(None))))
