"""One contract, every backend.

The same suite runs against a local directory always, and against an
S3-compatible store when `ATLAS_TEST_S3_ENDPOINT` is set (RustFS or MinIO from
`docker compose up rustfs`). A backend that passes local-only is not proven
S3-compatible; this is the test that says so.
"""

from __future__ import annotations

import datetime
import hashlib
import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from atlas_core.config import StorageSettings
from atlas_core.domain.errors import NotFoundError, StorageCapabilityError
from atlas_core.domain.ports import ContentStorePort
from atlas_storage.keys import content_key, staging_key
from atlas_storage.obstore_store import ObstoreContentStore, build_content_store

pytestmark = pytest.mark.asyncio


def _local(tmp_path: Path) -> ContentStorePort:
    settings = StorageSettings(backend="local", local_root=str(tmp_path))
    return build_content_store(settings)


def _s3() -> ContentStorePort:
    endpoint = os.environ.get("ATLAS_TEST_S3_ENDPOINT")
    if not endpoint:
        pytest.skip("ATLAS_TEST_S3_ENDPOINT is not set")
    settings = StorageSettings(
        backend="s3",
        bucket=os.environ.get("ATLAS_TEST_S3_BUCKET", "atlas"),
        endpoint=endpoint,
        region=os.environ.get("ATLAS_TEST_S3_REGION", "us-east-1"),
        access_key_id=os.environ.get("ATLAS_TEST_S3_ACCESS_KEY", "atlas"),
        secret_access_key=os.environ.get("ATLAS_TEST_S3_SECRET_KEY", "atlas-secret"),
        force_path_style=True,
        allow_http=endpoint.startswith("http://"),
    )
    return build_content_store(settings)


@pytest.fixture(params=["local", "s3"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> ContentStorePort:
    return _local(tmp_path) if request.param == "local" else _s3()


def _sha() -> str:
    return hashlib.sha256(uuid.uuid4().bytes).hexdigest()


def _key(tenant: str = "tenant-a") -> str:
    return content_key(tenant, _sha())


async def _chunks(*parts: bytes) -> AsyncIterator[bytes]:
    for part in parts:
        yield part


async def test_put_and_read_roundtrip(store: ContentStorePort) -> None:
    key = _key()
    body = b"# Atlas\n\nretrieval, measured.\n"

    stored = await store.put_bytes(key, body)

    assert stored.key == key
    assert stored.size_bytes == len(body)
    stream = await store.open_stream(key)
    assert b"".join([chunk async for chunk in stream]) == body


async def test_streaming_put_reports_the_bytes_it_wrote(store: ContentStorePort) -> None:
    key = staging_key(uuid.uuid4())

    stored = await store.put_stream(key, _chunks(b"one ", b"two ", b"three"))

    assert stored.size_bytes == len(b"one two three")
    stream = await store.open_stream(key)
    assert b"".join([chunk async for chunk in stream]) == b"one two three"


async def test_head_returns_metadata_and_none_for_a_missing_key(store: ContentStorePort) -> None:
    key = _key()
    await store.put_bytes(key, b"body")

    head = await store.head(key)

    assert head is not None
    assert head.key == key
    assert head.size_bytes == 4
    assert await store.head(_key()) is None


async def test_open_missing_key_raises_not_found(store: ContentStorePort) -> None:
    with pytest.raises(NotFoundError):
        await store.open_stream(_key())


async def test_copy_preserves_the_content(store: ContentStorePort) -> None:
    source = staging_key(uuid.uuid4())
    destination = _key()
    await store.put_bytes(source, b"copy me")

    await store.copy(source, destination)

    stream = await store.open_stream(destination)
    assert b"".join([chunk async for chunk in stream]) == b"copy me"


async def test_delete_is_idempotent(store: ContentStorePort) -> None:
    key = _key()
    await store.put_bytes(key, b"body")

    await store.delete(key)
    await store.delete(key)

    assert await store.head(key) is None


async def test_presign_is_reported_as_unavailable_on_local(tmp_path: Path) -> None:
    # A local directory cannot mint a signed URL. Failing loudly beats returning
    # a URL the browser cannot use.
    store = _local(tmp_path)
    assert isinstance(store, ObstoreContentStore)

    with pytest.raises(StorageCapabilityError):
        await store.presign_get(_key(), expires_in=datetime.timedelta(minutes=5))


async def test_presign_get_returns_a_url_when_the_backend_can_sign() -> None:
    store = _s3()
    key = _key()
    await store.put_bytes(key, b"body")

    url = await store.presign_get(key, expires_in=datetime.timedelta(minutes=5))

    assert url.startswith("http")
