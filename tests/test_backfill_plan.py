"""The backfill derives keys from bytes, and refuses bytes that do not match.

These are the two decisions the migration makes before it writes anything, so
they are tested without a database: a wrong key would put the wrong content
behind a document, and a trusted checksum would make the content address a lie.
"""

from __future__ import annotations

import hashlib

import pytest

from atlas_storage.keys import content_key
from scripts.migrate_to_versioned_storage import plan_version

BODY = b"# Atlas\n\nimmutable, versioned.\n"


def _checksum() -> str:
    return hashlib.sha256(BODY).hexdigest()


def test_plan_derives_the_key_from_the_bytes() -> None:
    plan = plan_version(
        tenant_id="tenant-a",
        checksum=_checksum(),
        mime="text/markdown",
        status="processing",
        chunk_count=3,
        content=BODY,
    )

    assert plan.sha256 == _checksum()
    assert plan.storage_key == content_key("tenant-a", _checksum())
    assert plan.size_bytes == len(BODY)


def test_plan_carries_the_document_state_into_version_one() -> None:
    plan = plan_version(
        tenant_id="tenant-a",
        checksum=_checksum(),
        mime="text/markdown",
        status="failed",
        chunk_count=0,
        content=BODY,
    )

    # The backfill mirrors current state; it does not infer it. A document that
    # is failed today becomes a failed version 1, which e03s03's serving
    # fallback then resolves.
    assert plan.seq == 1
    assert plan.status == "failed"
    assert plan.chunk_count == 0
    assert plan.checksum == _checksum()


def test_plan_rejects_bytes_that_do_not_match_the_stored_checksum() -> None:
    with pytest.raises(ValueError, match="checksum mismatch"):
        plan_version(
            tenant_id="tenant-a",
            checksum="0" * 64,
            mime="text/markdown",
            status="indexed",
            chunk_count=1,
            content=BODY,
        )


def test_plan_is_tenant_scoped() -> None:
    first = plan_version(
        tenant_id="tenant-a",
        checksum=_checksum(),
        mime="text/markdown",
        status="indexed",
        chunk_count=1,
        content=BODY,
    )
    second = plan_version(
        tenant_id="tenant-b",
        checksum=_checksum(),
        mime="text/markdown",
        status="indexed",
        chunk_count=1,
        content=BODY,
    )

    assert first.sha256 == second.sha256
    assert first.storage_key != second.storage_key
