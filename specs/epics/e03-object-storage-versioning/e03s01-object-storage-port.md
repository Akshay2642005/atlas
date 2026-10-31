# Story e03s01 — Object storage port with S3/RustFS, GCS, and local backends

**Epic:** e03 — Object storage + immutable document versions
**Risk:** P1 (new module, no behaviour change to existing paths)
**Depends on:** nothing
**Status:** done — evidence in `specs/verifications/BUILD-e03s01-object-storage-port.md`

## Why

Object storage is a hard requirement for the product, and it has to arrive before
the versioning migration so bytes are written to their final home once. Building a
Postgres content path now and migrating it later would move every document twice.

This story adds the port and the adapter, and wires them into DI. No existing
call path changes — `DocumentRepositoryPort` still owns content for the moment.

## Module zoom-out

- **`atlas_core.domain.ports`** — purpose: the set of interfaces the domain
  depends on. Callers: every adapter package, the pipeline, the API. Contracts:
  `rg "Port(Protocol)" src`. Adding `ContentStorePort` here keeps storage a
  dependency of the domain, not of an adapter.
- **New `atlas_storage`** — purpose: turn an object-storage SDK into that port.
  Callers: DI containers of `atlas_api` and `atlas_worker`. Contracts: the port
  only; nothing outside the package imports `obstore`.

## Requirements

**ADDED** — Content bytes are stored in an object store addressed by key.

- One async port covers S3-compatible stores (AWS S3, RustFS, MinIO), GCS, and a
  local directory used by tests and air-gapped dev.
- Keys are tenant-scoped and content-addressed: `{tenant}/{sha256[:2]}/{sha256}`.
  Staging keys live under `_staging/{uuid}` and are never content-addressed.
- Streaming put and streaming read; neither buffers the whole object in memory.
- `head` returns `None` for a missing key — absence is not an error.
- `delete` is idempotent.
- Presigning is a capability, not a guarantee: `local` raises a domain error
  instead of returning a broken URL.

## Implementation steps

1. Add `obstore` and a `StorageSettings` group (`backend`, `bucket`, `endpoint`,
   `region`, credentials, `allow_http`, `local_root`) → verify: `uv run python -c "from atlas_core.config import load_settings; load_settings().storage.backend"`
2. Add `StoredObject` to domain models and `ContentStorePort` to domain ports → verify: `uv run python -c "from atlas_core.domain.ports import ContentStorePort"`
3. Add `atlas_storage.keys` with content/staging key derivation and input validation → verify: `uv run pytest tests/test_storage_keys.py -q`
4. Add `atlas_storage.obstore_store.ObstoreContentStore` over `obstore` → verify: `uv run pytest tests/test_storage_contract.py -q`
5. Add `atlas_storage.providers.ObjectStorageProvider` and register it in both
   DI containers → verify: `uv run python -c "from atlas_api.container import build_container; from atlas_core.domain.ports import ContentStorePort; print('ok')"`
6. Add RustFS to `docker-compose.yml` and a `make storage-init` bucket-create target → verify: `grep -q rustfs docker-compose.yml && grep -q storage-init Makefile`
7. Contract-test the same suite against RustFS when `ATLAS_TEST_S3_ENDPOINT` is set, skipping otherwise → verify: `uv run pytest tests/test_storage_contract.py -q`
8. ADR 0005 and preflight → verify: `test -f specs/adr/0005-object-storage-and-content-addressing.md && make preflight`

## Verification script

```bash
uv run pytest tests/test_storage_keys.py tests/test_storage_contract.py -q
make preflight
# with RustFS running:
ATLAS_TEST_S3_ENDPOINT=http://localhost:9000 make storage-init
ATLAS_TEST_S3_ENDPOINT=http://localhost:9000 uv run pytest tests/test_storage_contract.py -q -m s3
```

## Out of scope

- Moving existing content out of `document_contents` (e03s02).
- Presigned upload endpoints (e03s04).
- Any retrieval change.

## Risks

- `obstore` is a new dependency; it is a Rust binding, so a wheel must exist for
  the platform. Verified on macOS arm64 and the `python:3.12-slim` Docker image.
- RustFS is S3-compatible but not S3; the contract suite is the gate that proves
  it, not the vendor claim.
