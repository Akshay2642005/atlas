"""Measure hosted rerank latency against candidate depth on the eval corpus.

Rerank cost scales with how many candidates are sent to the model, not with
`final_top_k`. This benchmark times the real hosted reranker for several
depths so the `RETRIEVAL__RERANK_TOP_K` default can be chosen from evidence.

    POSTGRES__URL=postgresql+asyncpg://atlas:atlas@localhost:5432/atlas_eval \
      uv run python scripts/bench_rerank_depth.py

Query vectors come from the local evaluation cache when present, so a rerun
does not depend on the embedder endpoint being warm.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from pathlib import Path

from atlas_core.config import AtlasSettings
from atlas_core.domain.ports import EmbedKind
from atlas_llm.nvidia_embed import NvidiaEmbedder
from atlas_llm.nvidia_rerank import NvidiaReranker
from atlas_retrieval.retriever import rrf_fusion, split_query
from atlas_vector.engine import create_engine_and_session
from atlas_vector.repositories import PgVectorRepository

DEFAULT_TENANT = "atlas-eval"
DEFAULT_URL = "postgresql+asyncpg://atlas:atlas@localhost:5432/atlas_eval"
DEFAULT_QUERIES = Path("specs/eval/queries-atlas-repo.jsonl")
DEFAULT_CACHE = Path(".eval-cache/embeddings.jsonl")
DEPTHS = (100, 50, 24, 12, 6)
REPS = 3


def _embed_key(model: str, dims: int, kind: str, text: str) -> str:
    return hashlib.sha256(f"{model}|{dims}|{kind}|{text}".encode()).hexdigest()


def _cached_vector(path: Path, key: str) -> list[float] | None:
    if not path.exists():
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        item = json.loads(raw)
        if str(item.get("key")) == key:
            return [float(value) for value in item["vector"]]
    return None


def _first_query(path: Path) -> str:
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            return str(json.loads(raw)["query"])
    raise SystemExit(f"{path} contains no queries")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--tenant", default=DEFAULT_TENANT)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--reps", type=int, default=REPS)
    args = parser.parse_args()

    settings = AtlasSettings()
    model = settings.nvidia.embedding_model

    query = _first_query(args.queries)
    sub_queries = split_query(query)
    text = sub_queries[0]
    vector = _cached_vector(args.cache, _embed_key(model, 4096, "query", text))
    if vector is None:
        print(f"cache miss for {text!r}; embedding live")
        embedder = NvidiaEmbedder(settings.nvidia)
        vector = (await embedder.embed([text], EmbedKind.QUERY))[0]
    print(f"query: {text}")

    postgres = settings.postgres.model_copy(update={"url": args.url})
    engine, session_factory = create_engine_and_session(postgres)
    vectors = PgVectorRepository(session_factory)
    try:
        dense = await vectors.semantic_search(args.tenant, vector, k=50)
        lexical = await vectors.lexical_search(args.tenant, text, k=50)
    finally:
        await engine.dispose()

    fused = rrf_fusion(
        [[str(chunk.id) for chunk, _ in dense], [str(chunk.id) for chunk, _ in lexical]],
        k=60,
    )
    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    texts = {str(chunk.id): chunk.text for chunk, _ in [*dense, *lexical]}
    documents = [texts[chunk_id] for chunk_id, _ in ordered]
    print(f"fused candidates: {len(documents)}")

    reranker = NvidiaReranker(settings.nvidia)
    print(f"\n{'depth':>5} {'median s':>9} {'min s':>7} {'max s':>7}")
    for depth in DEPTHS:
        samples: list[float] = []
        for _ in range(args.reps):
            started = time.perf_counter()
            scores = await reranker.rerank(text, documents[:depth])
            samples.append(time.perf_counter() - started)
            expected = min(depth, len(documents))
            if len(scores) != expected:
                raise SystemExit(f"expected {expected} scores at depth {depth}, got {len(scores)}")
        print(
            f"{depth:>5} {statistics.median(samples):>9.3f} "
            f"{min(samples):>7.3f} {max(samples):>7.3f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
