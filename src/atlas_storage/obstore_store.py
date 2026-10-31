"""`ContentStorePort` over `obstore` (S3, GCS, local).

`obstore` is a thin binding over Rust's `object_store`, so the same store API
covers AWS S3, RustFS, MinIO, and GCS. That is the whole reason it is the
dependency: without it, "S3-compatible plus GCS" means two SDKs with two
credential models and two async stories.
"""

from __future__ import annotations

import datetime
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal, cast

import obstore
from obstore.store import GCSStore, LocalStore, ObjectStore, S3Store

from atlas_core.config import StorageSettings
from atlas_core.domain.errors import (
    ConfigError,
    NotFoundError,
    StorageCapabilityError,
    StorageError,
)
from atlas_core.domain.models import StoredObject
from atlas_core.domain.ports import ContentStorePort

# `obstore` emits the builtin `FileNotFoundError` for a missing key on every
# backend; its own `NotFoundError` class is deprecated. One tuple, so a new
# backend cannot quietly fall into the "500 on a miss" path.
_NOT_FOUND: tuple[type[BaseException], ...] = (FileNotFoundError,)


def _validate_key(key: str) -> None:
    """Reject anything that could escape the bucket prefix or the local root."""
    if not key or key.startswith("/") or "\\" in key or ".." in key.split("/"):
        raise ValueError(f"invalid storage key: {key!r}")


class ObstoreContentStore(ContentStorePort):
    def __init__(
        self,
        store: ObjectStore,
        *,
        prefix: str = "",
        backend: str = "s3",
    ) -> None:
        self._store = store
        self._prefix = prefix.strip("/")
        self._backend = backend

    def _path(self, key: str) -> str:
        _validate_key(key)
        return f"{self._prefix}/{key}" if self._prefix else key

    async def put_bytes(self, key: str, data: bytes) -> StoredObject:
        try:
            result = await self._store.put_async(self._path(key), data)
        except Exception as exc:  # noqa: BLE001 — normalize every SDK failure
            raise StorageError(f"failed to store {key}", details={"key": key}) from exc
        return StoredObject(key=key, size_bytes=len(data), etag=_etag(result))

    async def put_stream(self, key: str, chunks: AsyncIterator[bytes]) -> StoredObject:
        size = 0

        async def counted() -> AsyncIterator[bytes]:
            nonlocal size
            async for chunk in chunks:
                size += len(chunk)
                yield chunk

        try:
            result = await self._store.put_async(self._path(key), counted())
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"failed to store {key}", details={"key": key}) from exc
        return StoredObject(key=key, size_bytes=size, etag=_etag(result))

    async def open_stream(self, key: str) -> AsyncIterator[bytes]:
        try:
            result = await self._store.get_async(self._path(key))
        except _NOT_FOUND as exc:
            raise NotFoundError(f"object not found: {key}", details={"key": key}) from exc
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"failed to read {key}", details={"key": key}) from exc

        async def chunks() -> AsyncIterator[bytes]:
            async for chunk in result:
                yield bytes(chunk)

        return chunks()

    async def head(self, key: str) -> StoredObject | None:
        try:
            meta = await self._store.head_async(self._path(key))
        except _NOT_FOUND:
            return None
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"failed to stat {key}", details={"key": key}) from exc
        return StoredObject(
            key=key,
            size_bytes=int(meta["size"]),
            etag=_etag(meta),
        )

    async def copy(self, source: str, destination: str) -> None:
        try:
            await self._store.copy_async(self._path(source), self._path(destination))
        except _NOT_FOUND as exc:
            raise NotFoundError(f"object not found: {source}", details={"key": source}) from exc
        except Exception as exc:  # noqa: BLE001
            raise StorageError(
                f"failed to copy {source} -> {destination}",
                details={"source": source, "destination": destination},
            ) from exc

    async def delete(self, key: str) -> None:
        try:
            await self._store.delete_async(self._path(key))
        except _NOT_FOUND:
            return
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"failed to delete {key}", details={"key": key}) from exc

    async def presign_put(self, key: str, *, expires_in: datetime.timedelta) -> str:
        return await self._presign("PUT", key, expires_in)

    async def presign_get(self, key: str, *, expires_in: datetime.timedelta) -> str:
        return await self._presign("GET", key, expires_in)

    async def _presign(
        self, method: Literal["GET", "PUT"], key: str, expires_in: datetime.timedelta
    ) -> str:
        if self._backend == "local":
            raise StorageCapabilityError(
                "the local storage backend cannot presign URLs; configure s3 or gcs"
            )
        try:
            # `obstore` types `sign_async` against the sign-capable stores; the
            # backend check above is what proves this one is. The cast keeps the
            # union out of the call site instead of silencing it with a blanket
            # `type: ignore`.
            signer = cast(Any, self._store)
            url = await obstore.sign_async(signer, method, self._path(key), expires_in)
            return str(url)
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"failed to presign {method} {key}", details={"key": key}) from exc


def _etag(result: Any) -> str | None:
    value = result.get("e_tag") if isinstance(result, dict) else None
    return str(value) if value is not None else None


def build_content_store(settings: StorageSettings) -> ContentStorePort:
    """Build the configured backend. Credentials come from settings or the SDK."""
    if settings.backend == "local":
        store: ObjectStore = LocalStore(prefix=Path(settings.local_root), mkdir=True)
        return ObstoreContentStore(store, backend="local")

    if not settings.bucket:
        raise ConfigError(f"storage.bucket is required for the {settings.backend} backend")

    if settings.backend == "gcs":
        gcs_store: ObjectStore = GCSStore.from_url(f"gs://{settings.bucket}")
        return ObstoreContentStore(gcs_store, prefix=settings.prefix, backend="gcs")

    kwargs: dict[str, Any] = {
        "region": settings.region,
        # Path style is what RustFS and MinIO speak by default; AWS is happier
        # with virtual-hosted style, so it is opt-in rather than assumed.
        "virtual_hosted_style_request": not settings.force_path_style,
    }
    if settings.endpoint:
        kwargs["endpoint"] = settings.endpoint
    if settings.access_key_id:
        kwargs["access_key_id"] = settings.access_key_id
    if settings.secret_access_key:
        kwargs["secret_access_key"] = settings.secret_access_key
    if settings.allow_http:
        kwargs["client_options"] = {"allow_http": True}
    s3_store: ObjectStore = S3Store(settings.bucket, **kwargs)
    return ObstoreContentStore(s3_store, prefix=settings.prefix, backend="s3")
