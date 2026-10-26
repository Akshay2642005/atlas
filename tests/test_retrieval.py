"""Retrieval fusion + query decomposition."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from atlas_core.config import RetrievalSettings
from atlas_core.domain.errors import InfraError
from atlas_core.domain.models import Chunk, Query
from atlas_core.domain.ports import EmbedKind
from atlas_retrieval.retriever import Retriever, rrf_fusion, split_query


class _Embedder:
    async def embed(self, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]:
        return [[0.0] for _ in texts]


class _Vectors:
    def __init__(self) -> None:
        self.chunk = Chunk(
            id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            tenant_id="tenant-a",
            seq=0,
            text="Atlas fuses ranks with RRF.",
        )

    async def semantic_search(
        self, tenant_id: str, embedding: Sequence[float], *, k: int
    ) -> list[tuple[Chunk, float]]:
        return [(self.chunk, 0.9)]

    async def lexical_search(
        self, tenant_id: str, text: str, *, k: int
    ) -> list[tuple[Chunk, float]]:
        return [(self.chunk, 0.4)]

    async def upsert_chunks(
        self,
        tenant_id: str,
        document_id: uuid.UUID,
        chunks: Sequence[tuple[Chunk, Sequence[float]]],
    ) -> int:
        return 0


class _FailingReranker:
    def __init__(self) -> None:
        self.calls = 0

    async def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls += 1
        raise InfraError("nvidia.rerank failed: [404] Not Found")


async def test_rerank_failure_pauses_retries_but_keeps_rrf_contexts() -> None:
    reranker = _FailingReranker()
    retriever = Retriever(
        RetrievalSettings(enable_rerank=True),
        _Vectors(),  # pyright: ignore[reportArgumentType] — structural fake
        _Embedder(),
        reranker,
    )
    query = Query(id=uuid.uuid4(), tenant_id="tenant-a", text="How does RRF work?")

    first = await retriever.retrieve(query)
    second = await retriever.retrieve(query)

    assert reranker.calls == 1  # second call is inside the cooldown, not retried
    assert [c.chunk_id for c in first.contexts] == [c.chunk_id for c in second.contexts]
    assert [(s.name, s.detail) for s in first.trace if s.name == "rerank"] == [
        ("rerank", "skipped (provider unavailable)")
    ]
    assert "rerank" not in [s.name for s in second.trace]


def test_rrf_prefers_documents_in_multiple_rankings() -> None:
    scores = rrf_fusion([["a", "b"], ["b", "c"]], k=60)
    assert scores["b"] > scores["a"] > scores["c"]
    assert scores["c"] > 0


def test_split_query_keeps_simple_queries_whole() -> None:
    assert split_query("What is our refund policy?") == ["What is our refund policy?"]


def test_split_query_decomposes_compound_questions() -> None:
    parts = split_query("What is the refund policy? How long does shipping take?")
    assert len(parts) == 2
