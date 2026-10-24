"""Hybrid retrieval: dense + lexical with Reciprocal-Rank Fusion and optional rerank."""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

import structlog

from atlas_core.config import RetrievalSettings
from atlas_core.domain.errors import InfraError
from atlas_core.domain.models import Query, TraceStep
from atlas_core.domain.ports import EmbedderPort, EmbedKind, RerankerPort, VectorRepositoryPort
from atlas_core.observability import get_tracer

_log = structlog.get_logger(__name__)
_tracer = get_tracer("atlas.retrieval")

_SPLIT_RE = re.compile(r"\?+|\band also\b|;|\n+")


@dataclass(frozen=True, slots=True)
class RetrievedContext:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    text: str
    final_score: float
    rrf_score: float
    rerank_score: float | None = None


@dataclass(frozen=True, slots=True)
class RetrievalOutcome:
    contexts: list[RetrievedContext]
    trace: list[TraceStep]


def split_query(text: str) -> list[str]:
    """Decompose compound questions; short/simple queries stay whole."""
    text = text.strip()
    if len(text) < 160 and text.count("?") <= 1:
        return [text]
    parts = [p.strip() for p in _SPLIT_RE.split(text) if p.strip()]
    return parts or [text]


def rrf_fusion(rankings: Sequence[Sequence[str]], *, k: int) -> dict[str, float]:
    """Reciprocal-rank fusion over ranked id lists (classic k=60)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


class Retriever:
    def __init__(
        self,
        settings: RetrievalSettings,
        vectors: VectorRepositoryPort,
        embedder: EmbedderPort,
        reranker: RerankerPort | None,
    ) -> None:
        self._settings = settings
        self._vectors = vectors
        self._embedder = embedder
        self._reranker = reranker if settings.enable_rerank else None

    async def retrieve(self, query: Query) -> RetrievalOutcome:
        with _tracer.start_as_current_span("retrieval.hybrid", attributes={"tenant.id": query.tenant_id}):
            subs = split_query(query.text)
            trace = [TraceStep("decompose", f"{len(subs)} sub-query(ies)")]
            semantic_ranks: list[list[str]] = []
            lexical_ranks: list[list[str]] = []
            by_id: dict[str, tuple[uuid.UUID, uuid.UUID, str]] = {}
            for sub in subs:
                embedding = (await self._embedder.embed([sub], kind=EmbedKind.QUERY))[0]
                dense = await self._vectors.semantic_search(query.tenant_id, embedding, k=self._settings.semantic_top_k)
                lexical = await self._vectors.lexical_search(query.tenant_id, sub, k=self._settings.lexical_top_k)
                semantic_ranks.append([str(c.id) for c, _ in dense])
                lexical_ranks.append([str(c.id) for c, _ in lexical])
                for chunk, _ in [*dense, *lexical]:
                    by_id.setdefault(
                        str(chunk.id), (chunk.id, chunk.document_id, chunk.text)
                    )
            fused = rrf_fusion([*semantic_ranks, *lexical_ranks], k=self._settings.rrf_k)
            trace.append(TraceStep("rrf", f"{len(fused)} unique chunks fused"))

            ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
            contexts: list[RetrievedContext] = []
            if self._reranker is not None and ordered:
                try:
                    scores = await self._reranker.rerank(query.text, [by_id[cid][2] for cid, _ in ordered])
                except InfraError as exc:
                    _log.warning("rerank.skipped", error=str(exc))
                    scores = None  # type: ignore[assignment]
                if scores is not None:
                    contexts = [
                        RetrievedContext(
                            chunk_id=by_id[cid][0],
                            document_id=by_id[cid][1],
                                text=by_id[cid][2],
                            rrf_score=rrf,
                            rerank_score=score,
                            final_score=score,
                        )
                        for (cid, rrf), score in zip(ordered, scores, strict=True)
                    ]
                    contexts.sort(key=lambda c: c.final_score, reverse=True)
                    trace.append(TraceStep("rerank", "nvidia rerank applied"))
            if not contexts:
                contexts = [
                    RetrievedContext(
                        chunk_id=by_id[cid][0],
                        document_id=by_id[cid][1],
                        text=by_id[cid][2],
                        final_score=rrf,
                        rrf_score=rrf,
                    )
                    for cid, rrf in ordered
                ]
            trace.append(TraceStep("select", f"top {min(len(contexts), self._settings.final_top_k)} contexts"))
            return RetrievalOutcome(contexts=contexts[: self._settings.final_top_k], trace=trace)
