"""Chunker strategies — the replaceable retrieval-granularity engine."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from atlas_core.config import ChunkingSettings
from atlas_core.domain.ports import ChunkerPort, EmbedderPort, EmbedKind

_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WS_RE = re.compile(r"\s+")


def approx_tokens(text: str) -> int:
    """Cheap deterministic token proxy (chars/4). Swap in tiktoken later if needed."""
    return max(1, (len(text) + 3) // 4)


def _norm_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _split_small(text: str) -> list[str]:
    """Split text into minimal units (paragraph -> sentence -> word) via separator cascade."""
    units: list[str] = [text]
    for sep in _SEPARATORS:
        nxt: list[str] = []
        for unit in units:
            nxt.extend(part for part in unit.split(sep) if part.strip())
        units = nxt
    return [_norm_ws(u) for u in units if u.strip()]


class RecursiveChunker(ChunkerPort):
    """Separator-hierarchy chunking, token-budgeted with word overlap (LangChain-style)."""

    def __init__(self, settings: ChunkingSettings) -> None:
        self._size = max(16, settings.chunk_size_tokens)
        self._overlap = min(max(0, settings.overlap_tokens), settings.chunk_size_tokens // 2)

    async def chunk(self, text: str) -> list[tuple[int, str]]:
        """Return deterministic chunks as (seq, text); token_count added by caller."""
        units = _split_small(text)
        chunks: list[str] = []
        cur: list[str] = []
        cur_len = 0
        for unit in units:
            unit_len = approx_tokens(unit)
            if cur_len + unit_len > self._size and cur:
                chunks.append(" ".join(cur))
                cur = cur[-self._overlap :] if self._overlap else []
                cur_len = sum(approx_tokens(u) for u in cur)
            cur.append(unit)
            cur_len += unit_len
        if cur:
            chunks.append(" ".join(cur))
        return [(seq, chunk) for seq, chunk in enumerate(chunks)]


class SemanticChunker(ChunkerPort):
    """Embedding-boundary chunking: pack sentences until a cosine-similarity valley.

    Needs one embed call per document — acceptable at ingest time. Falls back to
    empty when there are fewer than 2 sentences (recursion safety).
    """

    def __init__(self, embedder: EmbedderPort, settings: ChunkingSettings) -> None:
        self._embedder = embedder
        self._settings = settings

    async def chunk(self, text: str) -> list[tuple[int, str]]:
        sentences = [s for p in text.split("\n\n") for s in _SENTENCE_RE.split(p) if s.strip()]
        if len(sentences) < 2:
            return [(0, _norm_ws(text))] if text.strip() else []
        vectors = await self._embedder.embed(sentences, kind=EmbedKind.PASSAGE)
        dots = [_cos(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
        threshold = _percentile(dots, self._settings.semantic_percentile)
        groups: list[list[str]] = [[sentences[0]]]
        for i, dot in enumerate(dots):
            if dot >= threshold:
                groups[-1].append(sentences[i + 1])
            else:
                groups.append([sentences[i + 1]])
        out: list[tuple[int, str]] = []
        for seq, group in enumerate(groups):
            out.append((seq, " ".join(g.strip() for g in group)))
        return out


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 1.0
    index = min(len(ordered) - 1, max(0, int(q * len(ordered))))
    return ordered[index]
