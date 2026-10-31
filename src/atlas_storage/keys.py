"""Key derivation for the content store.

Two shapes, and the difference matters:

- **Content keys** are content-addressed: `{tenant}/{sha256[:2]}/{sha256}`. The
  same bytes always land on the same key, so a retry or a duplicate upload is a
  no-op instead of a second copy. The two-character shard keeps directory
  listings and S3 prefixes from growing into one flat bucket of millions.
- **Staging keys** are `_staging/{uuid}` and are *deliberately not* content
  addressed: the checksum is not known until the upload has finished streaming,
  so the bytes park under a random key and are copied into place on commit.

Tenant ids reach the key, which makes them an injection surface: a tenant of
`../../etc` would escape a local store's root. `_TENANT_RE` is the guard, and it
is intentionally narrower than "anything Postgres will accept".
"""

from __future__ import annotations

import re
import uuid

_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
# Matches the shape the API already hands out (`tenant-a`), and forbids path
# separators, leading dots, and uppercase-only tricks like "..".
_TENANT_RE = re.compile(r"\A[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}\Z")
STAGING_PREFIX = "_staging/"


def _validate_tenant(tenant_id: str) -> None:
    if not _TENANT_RE.match(tenant_id):
        raise ValueError(f"invalid tenant id for a storage key: {tenant_id!r}")


def content_key(tenant_id: str, sha256: str) -> str:
    """Return the immutable key for `sha256` bytes owned by `tenant_id`."""
    _validate_tenant(tenant_id)
    if not _SHA256_RE.match(sha256):
        raise ValueError(f"not a lowercase hex sha256: {sha256!r}")
    return f"{tenant_id}/{sha256[:2]}/{sha256}"


def staging_key(upload_id: uuid.UUID | str) -> str:
    """Return a throwaway key for an upload whose checksum is not known yet."""
    identifier = upload_id if isinstance(upload_id, uuid.UUID) else uuid.UUID(str(upload_id))
    return f"{STAGING_PREFIX}{identifier}"


def is_staging_key(key: str) -> bool:
    return key.startswith(STAGING_PREFIX)
