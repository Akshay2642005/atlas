# e01s01 — Ship 4096-dim embeddings and hosted Nemotron reranking

Status: done · BCPs: 8 · Recorded retroactively from the merged implementation.

## Context

The first end-to-end run against the live providers surfaced three gaps: the
Groq adapter sent requests to a doubled `/openai/v1` path, the embedding column
was 1024-dimensional while the requested model produces 4096, and reranking was
wired to a model the hosted account could not reach. The corpus also had to be
re-ingested because vectors of different widths are not interchangeable.

## Problem

1. `GROQ__BASE_URL` duplicated the path that `langchain-groq` already appends.
2. `nvidia/nemotron-3.5-embed-8b` returns 4096 dimensions, but the schema and
   client were configured for 1024.
3. `NVIDIARerank` derives `/ranking` from a deployment root, while the hosted
   Nemotron reranker is served from `/v1/retrieval/<model>/reranking`.

## Requirements

- **R1** — Embed chunks with `nvidia/nemotron-3.5-embed-8b` at 4096
  dimensions; the configured width must reach the client.
- **R2** — Store `vector(4096)`; purge incompatible vectors and mark their
  documents for re-ingestion.
- **R3** — Do not create an ANN index that pgvector cannot build for 4096-dim
  `vector`; keep the GIN full-text index.
- **R4** — Default reranking to `nvidia/nemotron-3.5-rerank-8b` on its hosted
  retrieval endpoint, with a self-hosted NIM/vLLM override.
- **R5** — Fail open on rerank errors: fall back to RRF and pause reranking for
  five minutes instead of retrying every request.
- **R6** — Let `langchain-groq` resolve its own endpoint; expose `BASE_URL`
  only for proxy deployments.
- **R7** — Run CI push checks on the actual default branch (`master`).

## Acceptance criteria

```gherkin
Scenario: 4096-dim embeddings are stored
  Given the NVIDIA embedder is configured with nemotron-3.5-embed-8b
  When a document is ingested
  Then every chunk has vector_dims = 4096
  And the client received dimensions = 4096

Scenario: pre-existing 1024-dim vectors are not silently reused
  Given the database contains 1024-dim chunks
  When migration 0002 runs
  Then those chunks are deleted
  And their documents are marked failed with a re-ingest message

Scenario: hosted reranking is applied
  Given the default NVIDIA rerank configuration
  When a chat query runs against an indexed document
  Then the trace contains "nvidia rerank applied"
  And the hosted request targets /v1/retrieval/nvidia/nemotron-3.5-rerank-8b/reranking

Scenario: rerank failure degrades gracefully
  Given the rerank provider raises InfraError
  When two queries run inside the cooldown window
  Then the first query logs rerank.skipped and returns RRF contexts
  And the second query does not call the reranker

Scenario: Groq uses its SDK endpoint
  Given GROQ__BASE_URL is unset
  When the Groq adapter is constructed
  Then ChatGroq.groq_api_base is None
```

## Design decisions

- Hosted model routing is registered through the package's public
  `register_model` hook instead of patching private client attributes.
- Rerank gets its own `RERANK_BASE_URL`; the OpenAI-compatible
  `NVIDIA__BASE_URL` is not reused because the hosted retrieval route differs.
- 4096 dimensions are kept over ANN indexing. See
  [ADR 0003](../../../docs/adr/0003-embedding-width-and-ann-indexing.md).

## Implementation summary

- `src/atlas_core/config.py` — 4096 embedding defaults, timeout, rerank model
  and rerank base URL, optional Groq base URL.
- `src/atlas_llm/nvidia_embed.py` — passes dimensions, widens the read
  timeout, retries transient failures once.
- `src/atlas_llm/nvidia_rerank.py` — hosted endpoint registration plus
  self-hosted override.
- `src/atlas_llm/nvidia_registry.py` — registers Nemotron models missing from
  the pinned LangChain table.
- `src/atlas_vector/schema.py` — `Vector(4096)`, no HNSW.
- `migrations/versions/0002_embedding_4096.py` — width change and re-ingest
  flags.
- `src/atlas_retrieval/retriever.py` — 300s rerank cooldown with RRF fallback.
- `tests/` — adapter, transport, and cooldown coverage.

## Verification evidence

- `make preflight`: ruff clean, mypy 51 files, 34 tests passed, no warnings.
- Live ingestion: 3 chunks, `min(vector_dims) = max(vector_dims) = 4096`.
- Live chat: trace reports `rerank: nvidia rerank applied`; the hosted
  `/reranking` probe returned HTTP 200 with ranking logits.
- No startup warnings after registering the Nemotron models.

## Risks and follow-ups

- Dense retrieval is an exact cosine scan, so latency grows with corpus size.
  A 10k–50k chunk benchmark should decide between keeping 4096 and moving to a
  2048-dim model with `halfvec` HNSW.
- The 8b embedder has high cold-start variance (observed 0.9–83s).
- Retrieval quality is not yet measured by an evaluation harness.

## Traceability

- `4737005` — embeddings, schema, migration, rerank wiring, Groq endpoint
- `e6f1b27` — rerank cooldown fallback
- `9dd3f0f` — console script rename
- `8619152` — stale pyright ignore removal
- `91c8c23` — CI push branch fix
