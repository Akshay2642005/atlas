# ADR 0001 — Retrieval, chunking, and DI decisions

Status: accepted · 2026-10-24

## Context

The platform must serve multi-tenant RAG with Groq chat, NVIDIA NIM embeddings, and pgVector, and stay deployable as two serverless-friendly units (API + ingestion worker). The brief asked for "semantic + recursive embedding" and "late embedding".

## Decisions

1. **No true late chunking with NVIDIA NIM.** The hosted embedding API returns pooled vectors only, and `nv-embedqa-e5-v5` has a 512-token context. Late chunking requires token-level embeddings, which the API does not expose. We use recursive chunking plus late *fusion* (RRF) and reranking. A token-level provider can be added as a new `EmbedderPort` adapter without touching the pipeline.
2. **Hybrid retrieval, fused late.** Dense cosine and Postgres `tsvector` rankings are computed independently, fused with RRF (k configurable), then optionally reranked by NVIDIA. This addresses vocabulary mismatch without a second vector store.
3. **Ports registered by protocol in Dishka.** Providers declare return types as `Protocol`s (`ChatPort`, `EmbedderPort`, `VectorRepositoryPort`, ...). Concrete adapters are invisible to consumers, which is what makes the provider/model/storage swaps additive.
4. **No dishka FastAPI integration.** It lags current FastAPI internals (0.162). FastAPI dependencies resolve from `app.state.container` with explicit adapters in `atlas_api/deps.py`; DI semantics stay unchanged.
5. **Queue via arq/Redis behind `QueuePort`.** Ingestion is asynchronous and retryable. Moving to SQS/Cloud Tasks/PGMQ means writing one adapter.
6. **Security is two-directional.** PII and prompt-injection checks run on ingest content and on model output; answers are masked after verification, and citations are filtered to actually-cited passages.
7. **Raw payloads live in `document_contents`.** Keeping bytes out of `documents` provides a clean lifecycle; move this table to object storage when retention policy requires it.

## Consequences

- Rerank endpoint/model are configuration, verified against current NIM docs at rollout time.
- Semantic chunking costs one embedding call per document; it is off by default.
- Hierarchical summaries (RAPTOR) and MMR are roadmap items; the `chunks.level` column and retrieval seam are prepared for them.
