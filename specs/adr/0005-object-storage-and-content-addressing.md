# ADR 0005 — Object storage for document bytes, addressed by content

Status: accepted · 2026-10-31

## Context

Document bytes live in `document_contents.content` (`bytea`) next to the
metadata they belong to. That was the right call while the only writer was the
evaluation harness and the largest document was a README.

It stops being right for a user-facing product:

- `POST /v1/documents` takes base64 JSON, so a 10 MB PDF arrives as a ~13 MB
  body, fully materialised in memory on both sides.
- Backups of the database grow with the corpus, and a restore has to move
  every document to recover one row.
- Serving an original document back to a user means streaming megabytes out of
  Postgres through the API, with no range requests and no CDN.
- The deployment story is self-hosted (see `specs/product/VISION_LATEST.yaml`),
  so "just use S3" is not available to every customer — the storage backend has
  to be a configuration choice, including a self-hosted S3-compatible one.

## Decision

**Content bytes live in an object store behind `ContentStorePort`.**

- One async port covers S3-compatible stores (AWS S3, **RustFS**, MinIO), GCS,
  and a local directory. The adapter is `obstore`, a binding over Rust's
  `object_store`; it is the only dependency that makes "S3-compatible plus GCS"
  one implementation instead of two SDKs with two credential models.
- **Keys are tenant-scoped and content-addressed:** `{tenant}/{sha256[:2]}/{sha256}`.
  The same bytes always land on the same key, which makes retries and duplicate
  uploads no-ops rather than second copies. Sharding on the first two hex
  characters keeps listings from becoming one flat prefix.
- **Dedupe is per tenant, never cross-tenant.** A shared blob table would let a
  tenant infer another tenant's content by comparing checksums. Storage is
  cheap; the leak is not.
- **Staging keys (`_staging/{uuid}`) are deliberately not content-addressed.**
  The checksum is not known until the upload finishes streaming, so bytes park
  at a random key and are copied into place on commit.
- **Presigning is a capability.** `local` cannot sign, and it raises
  `StorageCapabilityError` instead of returning a URL the browser cannot use.
- `local` is the default so a fresh clone needs no infrastructure. The compose
  stack sets `STORAGE__BACKEND=s3` against RustFS.

## Consequences

- The ingestion pipeline reads from the store instead of a `bytea` column. A
  loader that needs the whole payload (PDF text extraction) spills to a temp
  file above `STORAGE__MAX_INLINE_BYTES` instead of holding it in memory.
- Object storage introduces a failure Postgres never had: **orphans**. A commit
  that fails after the copy leaves a blob nobody references. The answer is a
  GC job plus the invariant "never delete a blob a live version points at",
  not a transaction — and it is written down here so it is not rediscovered.
- Tenant ids reach an object key, so they are validated against a narrow regex.
  Without it, a tenant named `../../etc` escapes a local store's root.
- Presigned PUT is single-shot. Resumable uploads (tus or S3 multipart) are
  deliberately out of scope until files are big enough to justify them; a
  single PUT with XHR progress covers the realistic sizes.
- Backups split in two: the database, and the bucket. They must be taken
  together or a restore produces metadata pointing at missing bytes.

## Alternatives considered

- **Keep bytes in Postgres.** Simplest, transactional, and still correct for
  small corpora. Rejected because the product needs original-document downloads
  and per-tenant storage growth that should not be a database migration.
- **boto3 + google-cloud-storage.** Two adapters, two auth models, sync clients
  wrapped in threads. Rejected for one `obstore` adapter.
- **Filesystem only (`local`).** Not viable for the self-hosted multi-node
  target, where two API replicas must see the same bytes.
- **Cross-tenant content addressing with refcounts.** Rejected: the existence
  oracle is a privacy regression in exchange for disk savings.
