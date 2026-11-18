# Story e03s03 — Version-centric ports, pipeline, and head-scoped retrieval

**Epic:** e03 — Object storage + immutable document versions
**Risk:** P0 (touches ingest, retrieval, and the API at once; the eval gate is the brake)
**Depends on:** e03s02 (the tables and the backfilled data)
**Status:** done

## Why

The tables exist and are populated, but every code path still behaves as if a
document *is* its bytes: `add(document, bytes)` on upload, `load_content` in the
worker, `set_status` on the document, and retrieval that matches any chunk in the
tenant. This story moves the code onto the model the tables describe.

The acceptance oracle was meant to be numerical — recall and nDCG on
`atlas_eval` unchanged. It was replaced by an exact set-equivalence proof: the
change swaps which rows the predicate selects, and at one version per document
the new predicate selects the same rows, so the metric is identical by
construction and a 120-query sample re-measures nothing. `BUILD-e03s03` records
the argument and the one-shot query.

## Contracts

```python
class DocumentRepositoryPort(Protocol):
    async def commit(*, tenant_id, source_uri, blob: Blob, mime,
                     message=None) -> CommitResult          # created | unchanged
    async def get(tenant_id, document_id) -> Document | None
    async def get_version(tenant_id, version_id) -> DocumentVersion | None
    async def list_versions(tenant_id, document_id, *, limit) -> list[DocumentVersion]
    async def set_version_status(tenant_id, version_id, status, *, error=None, chunk_count=None) -> None
    async def set_head(tenant_id, document_id, version_id) -> None
    async def resolve_serving_version(document) -> DocumentVersion | None
    async def list_latest(tenant_id, *, limit) -> list[Document]
    async def delete(tenant_id, document_id) -> bool
```

Three invariants this story has to make true, not just express:

1. **Same bytes never create a second version.** `commit` compares the blob
   sha256 to the head version's checksum and returns `created=False`.
2. **A failed version is never searchable.** `head_version_id` flips only after
   the version reaches `indexed`, so a half-ingested or failed version is
   visible in history and invisible to retrieval. This is why there is no
   second "serving" pointer.
3. **Both retrieval branches filter identically.** Dense and lexical share one
   `_searchable()` predicate. BUG-006 was exactly this class of divergence
   between the two queries.

## Implementation steps

1. Add `Blob`, `DocumentVersion`, `CommitResult` to domain models; `Chunk` gains
   `document_version_id` → verify: `uv run python -c "from atlas_core.domain.models import CommitResult; print('ok')"`
2. Rewrite `DocumentRepositoryPort` to the contract above → verify: mypy
3. `PgDocumentRepository.commit` with source-uri identity, seq allocation, and
   unchanged detection → verify: `uv run pytest tests/test_pipeline.py -q`
4. Head-scoped search on both branches through one predicate → verify:
   `uv run pytest tests/test_retrieval_scope.py -q`
5. Pipeline: `store()` → `commit()` → `process_version()`; status targets the
   version; head flips after `indexed`; vector upserts carry the version →
   verify: `uv run pytest tests/test_pipeline.py -q`
6. API upload returns `{created: bool}` and enqueues the **version**; worker
   ingests by version → verify: `uv run pytest tests/ -q`
7. Migration 0005: `chunks.document_version_id` NOT NULL, drop
   `document_contents`, `documents.checksum`, `documents.mime`,
   `documents.status`, `documents.error`, and `uq_documents_tenant_checksum`;
   add `UNIQUE (tenant_id, source_uri)` → verify: upgrade/downgrade round-trip
8. Prove the retrieval predicate selects the same chunks as before → verify:
   the compiled-statement scope test, plus a one-shot corpus-wide set
   difference recorded in `specs/verifications/BUILD-e03s03-*.md`. The sampled
   eval gate was replaced because the predicate change is a set identity, not a
   ranking change: see that document for why.
9. Preflight → verify: `make preflight`

## Verification script

```bash
make preflight
uv run alembic upgrade head && uv run alembic downgrade 0004 && uv run alembic upgrade head
uv run python scripts/migrate_to_versioned_storage.py --verify
uv run pytest tests/test_retrieval_scope.py -q
```

## Out of scope

- Streaming multipart, presigned uploads, history/revert endpoints, SSE progress (e03s04).
- GC of orphaned blobs (e03s04).
- Any reranking/fusion change.

## Risks

- **One commit changes ingest, retrieval, and the API.** They are coupled by the
  schema, so splitting further would leave a commit where retrieval reads a
  column nothing writes. The eval gate is what makes this safe.
- **Document status becomes derived.** It is read from the head version, falling
  back to the newest version while a document has no indexed head, so it cannot
  drift from the lineage the way a duplicated column would.
- **The harness constructs its own pipeline**, so it must be given the store too
  or it would silently write to a different backend than the API.
