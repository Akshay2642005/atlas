"""The ingestion pipeline: load -> sanitize -> guard -> PII-mask -> chunk -> embed -> index."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import structlog

from atlas_core.config import AtlasSettings
from atlas_core.domain.errors import DomainError, IngestError, PromptInjectionDetected
from atlas_core.domain.models import Chunk, Document, DocumentStatus
from atlas_core.domain.ports import (
    ChunkerPort,
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

_log = structlog.get_logger(__name__)
_tracer = get_tracer("atlas.ingest")


@dataclass(frozen=True, slots=True)
class IngestSpec:
    tenant_id: str
    content: bytes
    mime: str
    source_uri: str


@dataclass(frozen=True, slots=True)
class IngestResult:
    document: Document
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
    ) -> None:
        self._settings = settings
        self._documents = documents
        self._vectors = vectors
        self._embedder = embedder
        self._chunker = chunker
        self._sanitizer = sanitizer
        self._pii = pii
        self._guard = guard

    async def run(self, spec: IngestSpec) -> IngestResult:
        with _tracer.start_as_current_span("ingest.document", attributes={"tenant.id": spec.tenant_id}):
            document = Document(
                id=uuid.uuid4(),
                tenant_id=spec.tenant_id,
                source_uri=spec.source_uri,
                mime=spec.mime,
                status=DocumentStatus.PROCESSING,
                checksum=checksum(spec.content),
            )
            await self._documents.add(document, spec.content)
            return await self._process(document, spec.content)

    async def process_document(self, tenant_id: str, document_id: uuid.UUID) -> IngestResult:
        document = await self._documents.get(tenant_id, document_id)
        if document is None:
            raise DomainError(f"document {document_id} not found for tenant {tenant_id}")
        content = await self._documents.load_content(tenant_id, document_id)
        if content is None:
            raise IngestError(f"document {document_id} has no stored content")
        await self._documents.set_status(tenant_id, document_id, DocumentStatus.PROCESSING)
        return await self._process(document, content)

    async def _process(self, document: Document, raw: bytes) -> IngestResult:
        try:
            text = self._sanitizer.clean(load_bytes(raw, document.mime))

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
            count = await self._vectors.upsert_chunks(document.tenant_id, document.id, body)
            await self._documents.set_status(document.tenant_id, document.id, DocumentStatus.INDEXED)
            _log.info("ingest.indexed", document_id=str(document.id), chunks=count)
            return IngestResult(document=document, chunks=count)
        except Exception as exc:
            await self._documents.set_status(
                document.tenant_id, document.id, DocumentStatus.FAILED, error=str(exc)[:500]
            )
            _log.error("ingest.failed", document_id=str(document.id), error=str(exc))
            raise
