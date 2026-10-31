# Verification — e03s01 object storage port

Date: 2026-10-31 · Commit: pending

## What was proven

`ContentStorePort` works against a local directory **and** against RustFS, with
the same test suite, and presigned URLs minted by the adapter are usable by a
plain HTTP client (the browser upload path e03s04 depends on).

## Gates

```
make preflight
  ruff          All checks passed
  mypy          Success: no issues found in 56 source files
  basedpyright  0 errors, 0 warnings, 0 notes
  pytest        87 passed, 7 skipped
```

Local-only run (`ATLAS_TEST_S3_ENDPOINT` unset): the 7 S3 cases skip, 87 pass.

## Contract suite against RustFS

```
docker compose up -d rustfs
make storage-init
ATLAS_TEST_S3_ENDPOINT=http://localhost:9000 \
ATLAS_TEST_S3_ACCESS_KEY=atlas ATLAS_TEST_S3_SECRET_KEY=atlas-secret \
uv run pytest tests/test_storage_contract.py -q
→ 14 passed
```

The same 7 cases ran twice — once against `LocalStore`, once against RustFS —
covering bytes→bytes round-trip, streaming put with a byte count, `head`
metadata plus `None` on a miss, `NotFoundError` on `open_stream`, copy, delete
idempotency, and presign capability.

## Presigned URLs, end to end

```
presigned GET: 200 True
presigned PUT: 200
stored after PUT: 13
```

A presigned `PUT` wrote the object straight into RustFS, and a presigned `GET`
returned the original bytes through `urllib` — no credentials in the client.
That is the mechanism e03s04's browser upload relies on, so it is verified
before the endpoint exists rather than after.

## Notes

- `obstore` emits the builtin `FileNotFoundError` for a miss on every backend
  and its own `NotFoundError` class is deprecated; the adapter catches the
  builtin only.
- `LocalStore` cannot sign, so `presign_*` raises `StorageCapabilityError`
  (501) instead of returning a URL that cannot work.
- The `storage-init` service switched from `minio/mc` (no longer pullable) to
  `amazon/aws-cli` after the first run failed with `pull access denied`.
