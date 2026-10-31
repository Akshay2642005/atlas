# Story e03s02 — Content-addressed blobs and immutable document versions (schema + backfill)

**Epic:** e03 — Object storage + immutable document versions
**Risk:** P0 (irreversible data migration)
**Depends on:** e03s01 (the storage port the backfill writes through)
**Status:** done — evidence in `specs/verifications/BUILD-e03s02-versioned-schema-backfill.md`

## Why

`documents` is unique on `(tenant_id, checksum)` and holds the bytes in
`document_contents.content`. That identity is wrong for a product: the same
bytes twice is a 409, an edit is a duplicate document, and a failed ingest is
only recoverable through the delete-if-chunkless hack BUG-008 added.

This story lands the target shape — immutable content-addressed blobs, an
append-only version lineage, one mutable head pointer — and moves every
existing document into it, **without changing a single line of the read path**.
That is deliberate: the migration is the risky part, so it gets its own commit
where behaviour cannot drift, and the code that consumes the new tables arrives
in e03s03 with the eval harness as its oracle.

## Schema

```
blobs              id PK, tenant_id, sha256, size_bytes, storage_key, created_at
                   UNIQUE (tenant_id, sha256)
documents          + head_version_id FK -> document_versions.id  (NULL until committed)
document_versions  id PK, document_id FK, seq, blob_id FK, checksum, mime, message,
                   parent_version_id, status, error, chunk_count, created_at
                   UNIQUE (document_id, seq)
chunks             + document_version_id FK (nullable this story; NOT NULL in e03s03)
```

- `blobs` is keyed per tenant, never globally: a shared blob table would let a
  tenant infer another tenant's bytes by comparing checksums (ADR 0005).
- `document_versions.status` reuses `DocumentStatus`, so the domain enum does
  not fork into a second vocabulary.
- `chunks.document_version_id` is nullable here because the migration must run
  before the backfill; the backfill fills it, and e03s03 makes it `NOT NULL`
  once the code writes versions itself.

## Deviation from the plan (recorded, not silent)

The plan had `0004` also drop `uq_documents_tenant_checksum`. **Not done here.**
The API's 409 for "already ingested" is that constraint firing as an
`IntegrityError`; dropping it before the commit-semantics rewrite lands would
turn a 409 into a silent duplicate document. The constraint survives until
e03s03 replaces it with `commit() -> created | unchanged`, and `0005` (drop
`document_contents`, drop `documents.checksum`, drop the constraint) ships with
that code so no commit is ever red.

## Implementation steps

1. Add `BlobRow`, `DocumentVersionRow`, `documents.head_version_id`, and
   `chunks.document_version_id` to `atlas_vector.schema` → verify:
   `uv run python -c "from atlas_vector.schema import BlobRow, DocumentVersionRow; print('ok')"`
2. Write `0004_blobs_versions.py` (additive; no drops) → verify:
   `uv run alembic upgrade head && uv run alembic downgrade 0003 && uv run alembic upgrade head`
3. Write `scripts/migrate_to_versioned_storage.py`: for each document, stream
   `document_contents.content` through `ContentStorePort` to
   `{tenant}/{sha256[:2]}/{sha256}`, upsert a `blob`, insert
   `document_version(seq=1, status=<document.status>)`, set
   `documents.head_version_id`, repoint `chunks.document_version_id`.
   Idempotent, resumable, `--dry-run`, `--tenant` scoping → verify:
   `uv run pytest tests/test_backfill_plan.py -q`
4. `--verify` mode asserts: every document has a head, version seqs are dense
   from 1, every chunk carries a version, `count(chunks)` is unchanged, every
   head's blob exists in the store and rehashes to its key → verify: run it.
5. Run against `atlas` and `atlas_eval` (which holds both eval corpora by
   tenant) and record counts → verify: the verification doc shows the
   before/after chunk counts (9,102 unchanged).
6. Preflight → verify: `make preflight`

## Verification script

```bash
make migrate
uv run python scripts/migrate_to_versioned_storage.py --dry-run
uv run python scripts/migrate_to_versioned_storage.py
uv run python scripts/migrate_to_versioned_storage.py --verify
POSTGRES__URL=...atlas_eval uv run python scripts/migrate_to_versioned_storage.py --verify
make preflight
```

## Out of scope

- Any read-path change. Retrieval, ingest, and the API behave exactly as before.
- Dropping `document_contents` / `documents.checksum` / the unique constraint (e03s03).
- `commit()`, lineage branching beyond `seq=1`, revert (e03s03/e03s04).

## Risks

- **The migration touches real data.** It is idempotent and resumable, and
  `--verify` rehashes every head blob rather than trusting the insert.
- **Backfill is not atomic across storage and Postgres.** A crash between the
  object put and the row insert leaves an orphan blob; the GC job in e03s04
  collects it, and re-running the backfill reuses the same content key.
- **Local backend writes to `.atlas-storage/`**, which is gitignored; the
  verification run against the eval databases must record which backend it used.
