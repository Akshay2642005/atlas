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

A 4096-dimensional `vector` or `halfvec` column therefore cannot carry an ANN
index. The alternatives are all lossy or narrower:

- fewer dimensions (a native ≤2,000-dim model, or a Matryoshka truncation);
- `halfvec` ≤4,000 with a different model;
- a `bit` expression index via `binary_quantize(embedding)`, used only for
  candidate generation and followed by an exact cosine re-rank.

## Decision

Keep 4096 dimensions (`vector(4096)`), drop the `chunks_hnsw` index, and run
dense retrieval as an exact cosine scan. The GIN index on the generated `tsv`
column is unaffected, so hybrid retrieval still has an indexed lexical side.

Migration `0002` deletes pre-existing 1024-dimensional chunks, marks their
documents as needing re-ingestion, and alters the column. Re-ingesting is
required because vectors of different widths are not interchangeable.

## Consequences

- Recall is exact; there is no approximate-index tuning to get wrong.
- Dense latency grows linearly with a tenant's chunk count. This is acceptable
  for development and small tenants, not for a corpus of millions of chunks.
- If scale demands ANN, prefer the binary-quantized two-stage path over
  truncating embedding dimensions: it keeps the full vector for final scoring.
- The embedding width is configuration, not schema magic. Changing
  `NVIDIA__EMBEDDING_DIMENSIONS` requires a matching migration.
- `nvidia/nv-embedqa-e5-v5` is EOL (410 Gone since 2026-08-25); the default is
  the 4096-native `nvidia/nemotron-3.5-embed-8b`.
- The 8b embedder is the only 4096-capable model available on a typical NIM
  key. Its latency is variable (observed 0.9–83s, including 504s on cold
  start), so the adapter widens the client read timeout to 180s and retries
  transient failures once. A 2048-dim model is much faster but cannot satisfy
  a `vector(4096)` column.
- Reranking uses the hosted `nvidia/nemotron-3.5-rerank-8b` model. The hosted
  rerank API is served from `/v1/retrieval/<model>/reranking`, not the
  OpenAI-compatible `/v1` root, so the adapter registers that endpoint with
  `langchain-nvidia-ai-endpoints` instead of reusing `NVIDIA__BASE_URL`.
  Self-hosted NIM/vLLM deployments set `NVIDIA__RERANK_BASE_URL` and use the
  client's derived `/ranking` route.
- The retriever still treats rerank as optional: an unreachable endpoint is
  paused for five minutes instead of failing the request or retrying on every
  query.
