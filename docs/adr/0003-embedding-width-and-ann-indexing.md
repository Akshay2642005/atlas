# ADR 0003 — Embedding width and ANN indexing

Status: accepted · 2026-10-26

## Context

Atlas stores chunk embeddings in pgVector. The configured NVIDIA embedder
(`nvidia/nemotron-3.5-embed-8b`) produces 4096-dimensional vectors, and the
vector column and the embedder's `dimensions` setting must agree or inserts fail
at runtime.

pgvector 0.8 caps ANN indexes by width:

| storage type | max dimensions for HNSW / IVFFlat |
| --- | --- |
| `vector` | 2,000 |
| `halfvec` | 4,000 |
| `bit` | 64,000 |

A 4096-dimensional column therefore cannot carry an ANN index, whichever
storage type is used. The alternatives are all lossy or narrower:

- fewer dimensions (a native ≤2,000-dim model, or a Matryoshka truncation);
- a `bit` expression index via `binary_quantize(embedding)`, used only for
  candidate generation and followed by an exact cosine re-rank.

Three options were on the table:

1. keep 4096 dims and `vector(4096)`, exact scan (status quo at the time);
2. keep 4096 dims but store `halfvec(4096)`, exact scan (no re-embedding);
3. drop to 2048 dims so HNSW becomes possible.

## Measurements

`scripts/bench_indexing.py` (`make bench`) seeds synthetic clustered vectors
into a throwaway database and times server-side `EXPLAIN ANALYZE` execution for
100 queries, top-10. Full numbers: `specs/verifications/BENCH-indexing.md`.

At 200,000 chunks:

| setup | p50 / p95 | disk | HNSW build |
| --- | ---: | ---: | ---: |
| `vector(4096)` exact | 569 / 666 ms | 3.44 GB | — |
| `halfvec(4096)` exact | **355 / 461 ms** | **1.73 GB** | — |
| `halfvec(2048)` exact | 257 / 285 ms | 1.12 GB | — |
| `halfvec(2048)` + HNSW | **3.6 / 6.8 ms** | 2.76 GB (1.12 table + 1.64 index) | 113 s |

At 50,000 chunks `halfvec(4096)` is 140/163 ms; at 10,000 it is 15/16 ms.
HNSW recall@10 was 100% on the synthetic clusters, which is an optimistic upper
bound — real embeddings overlap far more, and recall already dipped to 97.9%
at 50k in an earlier run of the same harness.

Quality data from NVIDIA's published ViDoRe V3 text leaderboard:

| model | native dims | NDCG@10 | warm latency | cold latency |
| --- | ---: | ---: | ---: | ---: |
| `nemotron-3.5-embed-8b` | 4096 (2048 truncatable) | 60.65 | 0.87 s | 77–120 s+ |
| `nemotron-3.5-embed-1b` | 2048 | 57.80 | 0.60 s | ~7 s |

Both models were probed live: the 1B returns 2048 dims, and the 8B honours
`dimensions: 2048`.

## Decision

Keep 4096 dimensions and the 8B embedder. Store the column as
`halfvec(4096)` and keep exact cosine search; do not add HNSW.

Migration `0003` alters `chunks.embedding` from `vector(4096)` to
`halfvec(4096)` with `USING embedding::halfvec(4096)`. It is an in-place cast:
same values, half the bytes, no re-ingestion. Migration `0002` (1024 → 4096)
did require re-ingestion because that change corrupted the representation
itself, not merely its storage.

Migration `0002` dropped the stale `chunks_hnsw` index, which pgvector could
not have built at 4096 dims. The GIN index on the generated `tsv` column is
unaffected, so hybrid retrieval keeps an indexed lexical side.

## Why not 2048 + HNSW

HNSW is dramatically faster at query time — three orders of magnitude on this
data — but it is not free:

- **Disk:** 2.76 GB vs 1.73 GB for `halfvec(4096)`, a 59% increase. Lower
  dimension does not pay for the index.
- **Writes:** every insert rewires the graph and is much more expensive than an
  append. Ingestion is the workload Atlas actually spends wall-clock on.
- **Operations:** `m`, `ef_construction`, `ef_search`, and recall monitoring all
  become things that can be misconfigured or regress silently.
- **Quality:** 2048 dims either means a weaker model (~3 NDCG points for the
  1B) or a truncation whose retrieval cost is unmeasured on Atlas's corpus.
- **The latency it buys is not the bottleneck.** The exact scan at 50k chunks
  is 140 ms; the LLM round trip and embedding call are seconds. Saving 140 ms
  to add a lossy index is a bad trade at this scale.

## Consequences

- Recall is exact; there is no approximate-index tuning to get wrong.
- Dense latency still grows linearly with a tenant's chunk count. At 10k/50k
  chunks it is comfortably below interactive budgets; at 200k it is ~355 ms.
- If scale demands ANN, revisit in this order:
  1. HNSW over 8B truncated to 2048 dims (already supported by the model);
  2. `binary_quantize` candidate generation with exact re-rank, which keeps
     full precision for scoring;
  3. only then a smaller embedder.

  Trigger conditions: a tenant above ~500k–1M chunks, or retrieval p95 above
  ~1 s, or a UI that needs sub-10 ms retrieval.
- Any of those changes must be gated on an evaluation harness over Atlas's own
  corpus. The quality figures above are NVIDIA's benchmarks, not ours, and the
  recall figures come from synthetic clusters.
- The embedding width is configuration, not schema magic. Changing
  `NVIDIA__EMBEDDING_DIMENSIONS` requires a matching migration.
- `nvidia/nv-embedqa-e5-v5` is EOL (410 Gone since 2026-08-25); the default is
  the 4096-native `nvidia/nemotron-3.5-embed-8b`.
- The 8B's cold start is operational, not architectural: warm it is 0.87 s.
  Keep it warm (scheduled ping) or self-host the NIM rather than trading
  retrieval quality for a cold-start symptom. The adapter widens the client
  read timeout to 180s and retries transient failures once.
- Reranking uses the hosted `nvidia/nemotron-3.5-rerank-8b` model. The hosted
  rerank API is served from `/v1/retrieval/<model>/reranking`, not the
  OpenAI-compatible `/v1` root, so the adapter registers that endpoint with
  `langchain-nvidia-ai-endpoints` instead of reusing `NVIDIA__BASE_URL`.
  Self-hosted NIM/vLLM deployments set `NVIDIA__RERANK_BASE_URL` and use the
  client's derived `/ranking` route.
- The retriever still treats rerank as optional: an unreachable endpoint is
  paused for five minutes instead of failing the request or retrying on every
  query.
