"""The ingestion pipeline: store -> commit -> load -> guard -> chunk -> embed -> index."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import structlog

from atlas_core.config import AtlasSettings
from atlas_core.domain.errors import DomainError, IngestError, PromptInjectionDetected
from atlas_core.domain.models import (
    Blob,
    Chunk,
    CommitResult,
    Document,
    DocumentStatus,
    DocumentVersion,
)
from atlas_core.domain.ports import (
    ChunkerPort,
    ContentStorePort,
    DocumentRepositoryPort,
    EmbedderPort,
    EmbedKind,
    InjectionGuardPort,
    PiiDetectorPort,
    VectorRepositoryPort,
)
from atlas_core.observability import get_tracer
from atlas_core.security import Sanitizer, mask_text
from atlas_ingest.chunkers import approx_tokens
from atlas_ingest.loaders import checksum, load_bytes
from atlas_storage.keys import content_key

_log = structlog.get_logger(__name__)
_tracer = get_tracer("atlas.ingest")


@dataclass(frozen=True, slots=True)
class IngestSpec:
    tenant_id: str
    content: bytes
    mime: str
    source_uri: str
    message: str | None = None


@dataclass(frozen=True, slots=True)
class IngestResult:
    document: Document
    version: DocumentVersion
    chunks: int


class IngestPipeline:
    def __init__(  # noqa: PLR0913
        self,
        settings: AtlasSettings,
        documents: DocumentRepositoryPort,
        vectors: VectorRepositoryPort,
        embedder: EmbedderPort,
        chunker: ChunkerPort,
        sanitizer: Sanitizer,
        pii: PiiDetectorPort,
        guard: InjectionGuardPort,
        store: ContentStorePort,
    ) -> None:
        self._settings = settings
        self._documents = documents
        self._vectors = vectors
        self._embedder = embedder
        self._chunker = chunker
        self._sanitizer = sanitizer
        self._pii = pii
        self._guard = guard
        self._store = store

    async def store(self, spec: IngestSpec) -> CommitResult:
        """Park the bytes in the object store and record them as a version.

        The object write happens first and is idempotent on the content key, so
        a crash before the commit leaves an unreferenced blob (collectable)
        rather than a version pointing at bytes that were never written.
        """
        digest = checksum(spec.content)
        key = content_key(spec.tenant_id, digest)
        if await self._store.head(key) is None:
            await self._store.put_bytes(key, spec.content)
        blob = Blob(sha256=digest, size_bytes=len(spec.content), storage_key=key)
        return await self._documents.commit(
            tenant_id=spec.tenant_id,
            source_uri=spec.source_uri,
            blob=blob,
            mime=spec.mime,
            message=spec.message,
        )

    async def run(self, spec: IngestSpec) -> IngestResult:
        with _tracer.start_as_current_span(
            "ingest.document", attributes={"tenant.id": spec.tenant_id}
        ):
            committed = await self.store(spec)
            if committed.version.status is DocumentStatus.INDEXED:
                # Same bytes as the version already serving: no work, and no
                # second version. This is the "200 {unchanged: true}" path.
                return IngestResult(
                    document=committed.document,
                    version=committed.version,
                    chunks=committed.version.chunk_count,
                )
            return await self.process_version(spec.tenant_id, committed.version.id)

    async def process_version(self, tenant_id: str, version_id: uuid.UUID) -> IngestResult:
        """Ingest one version. Idempotent, and safe to retry after a failure."""
        version = await self._documents.get_version(tenant_id, version_id)
        if version is None:
            raise DomainError(f"version {version_id} not found for tenant {tenant_id}")
        document = await self._documents.get(tenant_id, version.document_id)
        if document is None:
            raise DomainError(f"document {version.document_id} not found for tenant {tenant_id}")

        await self._documents.set_version_status(tenant_id, version_id, DocumentStatus.PROCESSING)
        raw = await self._read(version.storage_key)
        return await self._process(document, version, raw)

    async def _read(self, storage_key: str) -> bytes:
        if not storage_key:
            raise IngestError("version has no blob; the commit did not record one")
        stream = await self._store.open_stream(storage_key)
        # The loaders take bytes today (PDF text extraction needs the whole
        # payload anyway); streaming loaders land with multipart upload.
        return b"".join([chunk async for chunk in stream])

    async def _process(
        self, document: Document, version: DocumentVersion, raw: bytes
    ) -> IngestResult:
        try:
            text = self._sanitizer.clean(load_bytes(raw, version.mime))

            verdict = self._guard.scan(text)
            if verdict.risky:
                raise PromptInjectionDetected(
                    "prompt-injection patterns detected in source content",
                    details={"patterns": verdict.matches},
                )

            findings = self._pii.detect(text)
            if self._settings.security.pii_policy == "mask" and findings:
                text = mask_text(text, findings)
                _log.info("pii.masked", kinds=sorted({f.kind for f in findings}))

            pieces = await self._chunker.chunk(text)
            if not pieces:
                raise IngestError("content produced zero chunks")

            embeddings = await self._embedder.embed([p[1] for p in pieces], kind=EmbedKind.PASSAGE)
            body = [
                (
                    Chunk(
                        id=uuid.uuid4(),
                        document_id=document.id,
                        document_version_id=version.id,
                        tenant_id=document.tenant_id,
                        seq=seq,
                        level=0,
                        text=chunk_text,
                        token_count=approx_tokens(chunk_text),
                    ),
                    embedding,
                )
                for (seq, chunk_text), embedding in zip(pieces, embeddings, strict=True)
            ]
            count = await self._vectors.upsert_chunks(document.tenant_id, version.id, body)
            await self._documents.set_version_status(
                document.tenant_id, version.id, DocumentStatus.INDEXED, chunk_count=count
            )
            # The head moves last, and only now: until this line the previous
            # version keeps serving, and a failure above never becomes visible
            # to retrieval.
            await self._documents.set_head(document.tenant_id, document.id, version.id)
            _log.info("ingest.indexed", document_id=str(document.id), chunks=count)
            return IngestResult(
                document=document,
                version=version,
                chunks=count,
            )
        except Exception as exc:
            await self._documents.set_version_status(
                document.tenant_id, version.id, DocumentStatus.FAILED, error=str(exc)[:500]
            )
            _log.error("ingest.failed", document_id=str(document.id), error=str(exc))
            raise
