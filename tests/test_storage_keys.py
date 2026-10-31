"""Storage keys are tenant-scoped, content-addressed, and traversal-proof."""

from __future__ import annotations

import uuid

import pytest

from atlas_storage.keys import content_key, is_staging_key, staging_key

SHA = "19cebadeeb8d3181d01ef00399674ec41bd2f0363850cdccc7685d4260c5a1c6"


def test_content_key_is_content_addressed_and_sharded() -> None:
    assert content_key("tenant-a", SHA) == f"tenant-a/19/{SHA}"


def test_content_key_is_stable_for_the_same_bytes() -> None:
    # This is the property the whole retry story rests on: same bytes, same key,
    # so a second upload cannot create a second copy.
    assert content_key("tenant-a", SHA) == content_key("tenant-a", SHA)


def test_content_key_separates_tenants() -> None:
    assert content_key("tenant-a", SHA) != content_key("tenant-b", SHA)


@pytest.mark.parametrize("tenant", ["../etc", "a/b", "", ".hidden", "tenant a"])
def test_content_key_rejects_an_unsafe_tenant(tenant: str) -> None:
    with pytest.raises(ValueError):
        content_key(tenant, SHA)


@pytest.mark.parametrize("sha", ["abc", SHA.upper(), f"{SHA}0", "../" + SHA])
def test_content_key_rejects_a_non_sha256(sha: str) -> None:
    with pytest.raises(ValueError):
        content_key("tenant-a", sha)


def test_staging_key_is_derived_from_the_upload_id() -> None:
    upload_id = uuid.uuid4()

    assert staging_key(upload_id) == f"_staging/{upload_id}"
    assert is_staging_key(staging_key(upload_id))
    assert not is_staging_key(content_key("tenant-a", SHA))
