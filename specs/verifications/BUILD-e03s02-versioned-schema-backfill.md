# Verification — e03s02 versioned schema and backfill

Date: 2026-10-31 · Commit: pending

## What was proven

Every document in every local database now has a content-addressed blob in the
object store, an immutable version 1, a head pointer, and chunks that carry
their version id — with **the chunk count unchanged**. Nothing on the read path
was touched, so retrieval cannot drift in this commit; the head-scoped join and
its eval gate belong to e03s03.

## Gates

```
make preflight
  ruff          All checks passed
  mypy          Success: no issues found in 56 source files
  basedpyright  0 errors, 0 warnings, 0 notes
  pytest        91 passed, 7 skipped

uv run mypy scripts/migrate_to_versioned_storage.py   Success
uv run ruff check scripts/migrate_to_versioned_storage.py   All checks passed
```

## Migration 0004 round-trips

```
uv run alembic current       0003
uv run alembic upgrade head  0004 (head)
uv run alembic downgrade 0003
uv run alembic upgrade head  0004 (head)
```

The downgrade drops `chunks.document_version_id`, the head FK and column, then
`document_versions` and `blobs` — in that order, so the cycle between
`documents` and `document_versions` never blocks it.

## Backfill

`atlas` (dev database):

```
dry-run:  documents=3   written=3   skipped=0 problems=0
backfill: documents=3   written=3   skipped=0 chunks_linked=5   problems=0
verify:   documents=3   written=0   skipped=0 chunks_linked=5   problems=0
```

`atlas_eval` holds both evaluation corpora by tenant (`atlas-eval`: 20 docs /
112 chunks, `atlas-eval-docs`: 929 docs / 8,990 chunks — there is no separate
`atlas_eval_docs` database):

```
dry-run:  documents=949 written=949 skipped=0 chunks_linked=0    problems=0
backfill: documents=949 written=949 skipped=0 chunks_linked=9102 problems=0
verify:   documents=949 written=0   skipped=0 chunks_linked=9102 problems=0
```

Chunk count before: **9102** · after: **9102**.

`--verify` does not trust the insert. It asserts:

- no document is missing a head version;
- no chunk has a null `document_version_id`;
- no chunk points at a version belonging to a different document;
- `count(chunks)` equals the sum of `document_versions.chunk_count`;
- every head's blob exists, its size matches the row, and **streaming it back
  rehashes to the key** — all 949 blobs, not a sample.

## Idempotence

```
re-run backfill: documents=0 written=0 skipped=949 problems=0
document_versions: 949   blobs: 949   (unchanged after the re-run)
```

A second pass writes nothing. The content key comes from the bytes, the blob
unique key is `(tenant_id, sha256)`, and the version is `(document_id, seq)`, so
a crash mid-run is resumed by running the same command again rather than
cleaned up by hand.

## State carried over faithfully

```
document_versions by status: indexed | 949
head version status != document status: 0
```

The backfill mirrors state; it does not infer it. A document that is `failed`
today would have become a failed version 1, which e03s03's serving fallback
resolves to the newest indexed ancestor.

## Deviation from the plan

`0004` is additive only. The plan had it also drop
`uq_documents_tenant_checksum`; that constraint is what raises the API's 409 for
duplicate content, and dropping it before the commit-semantics rewrite would
turn a 409 into a silent duplicate document. It is dropped in `0005` together
with `document_contents` and `documents.checksum`, in e03s03, alongside the code
that stops depending on them.

## Notes

- The local backend wrote to `.atlas-storage/` (gitignored), so this run
  exercised `LocalStore`. RustFS was proven separately in e03s01; the backfill
  uses the same port, and the S3 path is covered by the contract suite.
- Backfill order is object first, then metadata. A crash in between leaves an
  orphan blob the GC job reclaims; the reverse order would leave a row pointing
  at bytes that do not exist, which nothing can repair.
