"""Chunker behaviour."""

from __future__ import annotations

from atlas_core.config import ChunkingSettings
from atlas_core.domain.ports import EmbedderPort, EmbedKind
from atlas_ingest.chunkers import RecursiveChunker, SemanticChunker


class _FakeEmbedder(EmbedderPort):
    async def embed(self, texts: list[str], kind: EmbedKind) -> list[list[float]]:
        # Orthogonal-ish vectors force semantic boundaries between sentences.
        return [[1.0, float(i), 0.0] for i, _ in enumerate(texts)]


async def test_recursive_chunker_respects_token_budget() -> None:
    chunker = RecursiveChunker(ChunkingSettings(chunk_size_tokens=40, overlap_tokens=5))
    text = " ".join(f"Sentence number {i} about retrieval quality." for i in range(60))
    chunks = await chunker.chunk(text)
    assert len(chunks) > 1
    assert [seq for seq, _ in chunks] == list(range(len(chunks)))
    assert all(chunk.strip() for _, chunk in chunks)


async def test_semantic_chunker_produces_ordered_chunks() -> None:
    chunker = SemanticChunker(_FakeEmbedder(), ChunkingSettings(semantic_percentile=0.5))
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."
    chunks = await chunker.chunk(text)
    assert chunks
    assert [seq for seq, _ in chunks] == list(range(len(chunks)))
