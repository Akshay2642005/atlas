"""Dishka provider for the ingestion pipeline."""

from __future__ import annotations

from dishka import Provider, Scope, provide

from atlas_core.config import AtlasSettings
from atlas_core.domain.ports import (
    ChunkerPort,
    DocumentRepositoryPort,
    EmbedderPort,
    InjectionGuardPort,
    PiiDetectorPort,
    VectorRepositoryPort,
)
from atlas_core.security import Sanitizer
from atlas_ingest.chunkers import RecursiveChunker, SemanticChunker
from atlas_ingest.pipeline import IngestPipeline


class IngestProvider(Provider):
    @provide(scope=Scope.APP)
    def chunker(self, settings: AtlasSettings, embedder: EmbedderPort) -> ChunkerPort:
        if settings.chunking.enable_semantic_chunks:
            return SemanticChunker(embedder, settings.chunking)
        return RecursiveChunker(settings.chunking)

    @provide(scope=Scope.APP)
    def pipeline(  # noqa: PLR0913
        self,
        settings: AtlasSettings,
        documents: DocumentRepositoryPort,
        vectors: VectorRepositoryPort,
        embedder: EmbedderPort,
        chunker: ChunkerPort,
        sanitizer: Sanitizer,
        pii: PiiDetectorPort,
        guard: InjectionGuardPort,
    ) -> IngestPipeline:
        return IngestPipeline(
            settings=settings,
            documents=documents,
            vectors=vectors,
            embedder=embedder,
            chunker=chunker,
            sanitizer=sanitizer,
            pii=pii,
            guard=guard,
        )


def providers() -> list[Provider]:
    return [IngestProvider()]
