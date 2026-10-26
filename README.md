# Atlas

Strictly-typed, DI-first RAG platform for production: **FastAPI + LangChain providers + pgVector**, with a separate ingestion worker. Chat runs through LangChain (`langchain-groq`, `langchain-openai`, `langchain-anthropic`, `langchain-nvidia-ai-endpoints`) behind one `ChatPort`; embeddings and reranking use the NVIDIA NIM LangChain package. Named after the map of the world — the retrieval layer is a tenant-isolated index over your corpus.

## Why this shape (design gaps closed)

The original brief asked for "semantic + recursive embedding" and "late embedding". Three corrections that made the system coherent:

1. **Late chunking is not available on NVIDIA NIM embeddings.** NIM embedding APIs return one pooled vector per input (and `nv-embedqa-*` models have a 512-token window), so token-level pooling — what "late chunking" actually requires — is impossible through that API. Atlas therefore uses **hierarchical recursive chunking** (level-aware `chunks.level`, parent/child ready) plus **late interaction at retrieval time**: dense + lexical candidates are fused *after* retrieval (RRF) and reranked against the untouched query. Swapping in a token-embedding provider (e.g. Jina v3) later only requires a new `EmbedderPort` adapter.
2. **"Semantic" and "recursive" are chunking strategies, not embedding models.** Atlas ships both behind `ChunkerPort`: `RecursiveChunker` (default, zero extra API cost) and `SemanticChunker` (embedding-similarity boundaries; one embed call per document at ingest).
3. **Dense-only retrieval is not production retrieval.** Atlas does **hybrid search**: pgVector cosine + Postgres full-text (`tsvector`/`ts_rank_cd`), fused with Reciprocal Rank Fusion, then NVIDIA rerank, then a token budget. Citations are verified against retrieved chunk ids before they reach the client.

Also included because enterprise RAG needs them and the brief implied them: two-direction PII policy (ingest masking + answer masking), prompt-injection detection on **both** source content and model output, rate limiting per tenant, security headers, request ids, typed error envelopes, chaos-based fault injection for resilience tests, OpenTelemetry traces, structured logs, readiness probes, tenant isolation on every SQL predicate, and queue-based async ingestion.

## Layout

```
src/
  atlas_core/        domain models, ports (Protocols), config, security, observability, DI providers
  atlas_llm/         LangChain adapters: chat (Groq/OpenAI/Anthropic/NVIDIA), NVIDIA embeddings + rerank
  atlas_vector/      SQLAlchemy async schema, pgVector repositories, Dishka provider
  atlas_ingest/      loaders (txt/md/html/pdf), chunkers, ingest pipeline, provider
  atlas_retrieval/   hybrid retriever (RRF/rerank), grounded AnswerService (citations + trace)
  atlas_api/         FastAPI app, auth/rate limit deps, middlewares, routes, schemas
  atlas_worker/      arq worker + ingestion tasks
tests/               unit + API + pipeline tests (no network, no DB)
migrations/          Alembic (pgvector extension, GIN full-text index)
```

Dependencies point inward: `atlas_core` knows nothing about FastAPI, LangChain, or Postgres. Everything crosses the boundary through `Protocol` ports, registered in Dishka **by port type** (protocols, not concrete classes).

## Quickstart

```bash
cp .env.example .env          # then set GROQ__API_KEY and NVIDIA__API_KEY (or another CHAT__PROVIDER)
uv sync
docker compose up -d postgres redis
uv run alembic upgrade head

uv run atlas-api                            # http://localhost:8000/docs
uv run arq atlas_worker.main.WorkerSettings # separate terminal
```

> `.env` in this repo predates the settings schema; if it still uses `GROQ_API_KEY` / `NVIDIA_API_KEY`, rename them to `GROQ__API_KEY` / `NVIDIA__API_KEY` (nested groups use a double underscore).

Ingest and ask:

```bash
curl -s localhost:8000/v1/documents \
  -H "X-API-Key: dev-key" -H "Content-Type: application/json" \
  -d '{"source_uri":"handbook.md","mime":"text/markdown","content":"# Refunds\nRefunds are processed in 5 business days."}'

curl -s localhost:8000/v1/chat \
  -H "X-API-Key: dev-key" -H "Content-Type: application/json" \
  -d '{"query":"How long do refunds take?"}'
```

Streaming: `{"query": "...", "stream": true}` → SSE with `delta` events and one `final` event carrying citations + trace.

## API

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/healthz` | none | liveness |
| GET | `/readyz` | none | DB + Redis readiness |
| POST | `/v1/chat` | API key | grounded answer / SSE stream |
| POST | `/v1/documents` | API key | register document + enqueue ingest (202) |
| GET | `/v1/documents` | API key | latest documents for the tenant |
| GET | `/v1/documents/{id}` | API key | document status |
| DELETE | `/v1/documents/{id}` | API key | delete document + chunks (FK cascade) |

Auth: `X-API-Key: <key>` or `Authorization: Bearer <key>`. Keys map to tenants via `SECURITY__API_KEYS="key=tenant,key2=tenant2"`. Rate limit is a per-key sliding window (`SECURITY__RATE_LIMIT_RPM`).

Errors are uniform: `{"error": {"code", "message", "details"}}`; 429 adds `Retry-After`.

## Configuration

All settings are typed in `atlas_core/config.py`; nested env groups use `__`:

| Group | Notable fields |
|---|---|
| `CHAT__` | `PROVIDER` (`groq\|openai\|anthropic\|nvidia`, default `groq`), `TEMPERATURE`, `MAX_TOKENS`, `REQUEST_TIMEOUT`, `MAX_RETRIES` |
| `GROQ__` | `API_KEY`, `MODEL` (default `llama-3.3-70b-versatile`), optional `BASE_URL` (proxy only; the SDK supplies its own) |
| `OPENAI__` | `API_KEY`, `MODEL` (`gpt-4o-mini`), `BASE_URL` — set this to any OpenAI-compatible `/v1` root |
| `ANTHROPIC__` | `API_KEY`, `MODEL` (`claude-haiku-4-5-20251001`), `BASE_URL` |
| `NVIDIA__` | `API_KEY`, `CHAT_MODEL` (`meta/llama-3.1-8b-instruct`), `EMBEDDING_MODEL` (`nvidia/nemotron-3.5-embed-8b`, 4096 dims), `EMBEDDING_DIMENSIONS` (default `4096`), `RERANK_MODEL` (`nvidia/nemotron-3.5-rerank-8b`), `RERANK_BASE_URL` (self-hosted NIM/vLLM only) |
| `POSTGRES__` | `URL` (asyncpg), pool + statement timeout |
| `REDIS__` | queue + rate limiter |
| `CHUNKING__` | `CHUNK_SIZE_TOKENS`, `OVERLAP_TOKENS`, `ENABLE_SEMANTIC_CHUNKS` |
| `RETRIEVAL__` | `SEMANTIC_TOP_K`, `LEXICAL_TOP_K`, `RERANK_TOP_K` (candidates sent to the reranker, default 50), `FINAL_TOP_K`, `ENABLE_RERANK`, `RRF_K` |
| `SECURITY__` | `API_KEYS`, `PII_POLICY` (`off|mask`), `RATE_LIMIT_RPM`, `DISABLE_INJECTION_GUARD` |
| `OBSERVABILITY__` | `LOG_LEVEL`, `LOG_JSON`, `OTLP_ENDPOINT` (empty = tracing off) |
| `CHAOS__` | `ENABLED`, `FAILURE_RATE`, `SLEEP_SECONDS` — never enable in prod |

### Embedding width and indexing

Embeddings are 4096-dimensional, stored as `halfvec(4096)`. pgvector caps ANN indexes at 4,000 dimensions for `halfvec`, so Atlas does **not** create an HNSW/IVFFlat index: dense retrieval runs an exact cosine scan, while lexical retrieval stays GIN-indexed. `halfvec` halves storage and exact-scan I/O versus `vector(4096)` with no re-embedding. That is exact but scales linearly with each tenant's chunk count; see [ADR 0003](specs/adr/0003-embedding-width-and-ann-indexing.md) and [the benchmark](specs/verifications/BENCH-indexing.md) for the measured alternatives (a ≤2,000-dim model, or a binary-quantized expression index). The 4096-native `nemotron-3.5-embed-8b` is the only 4096-dim model on the NVIDIA key used here and has variable cold-start latency; the adapter allows 180s and retries transient failures once.

### Retrieval evaluation

The decision gate for embedding-width or model changes is `scripts/eval_retrieval.py`, not an ad-hoc benchmark. It ingests a corpus, generates one judged question per sampled chunk, and compares dense arms on recall@k, MRR, nDCG@k, and latency with a paired bootstrap confidence interval.

```bash
make eval-db        # create atlas_eval and migrate it
make eval-ingest    # ingest specs + README into the atlas-eval tenant
make eval-queries   # ask the chat model for one question per sampled chunk
make eval-warm      # cache query embeddings first (the 8B embedder is flaky)
make eval           # full4096 vs trunc2048 -> specs/verifications/EVAL-retrieval.md
make eval-pipeline  # dense vs RRF vs RRF+rerank -> EVAL-retrieval-hybrid.md
```

Supported arms are `full4096`, `trunc2048`, `trunc2048-hnsw`, and `reembed:<model>@<dims>[:hnsw]` for a real model swap. `hybrid` runs the production retriever (dense + lexical with reciprocal-rank fusion) and `hybrid-rerank` adds the hosted reranker, so those two measure the pipeline users actually hit, including its latency. Ground truth is stored as `(document checksum, chunk seq)`, so a query set survives re-ingestion and can be committed. The dense arms are deliberately fusion-free: mixing them with reranking would hide the embedding change being tested.

Reranking uses the hosted `Nemotron-3.5-Rerank-8B` retrieval endpoint by default (`RERANK_BASE_URL` empty). It fails open: if the NIM account cannot reach the model, the retriever logs one warning and falls back to RRF order for five minutes instead of retrying on every request. Set `RETRIEVAL__ENABLE_RERANK=false` to skip it entirely.

`make eval-warm` fills `.eval-cache/embeddings.jsonl` with every query and sub-query vector a run needs, retrying a cold 8B endpoint with backoff. Run it in a loop if the endpoint is returning 504s; `run` then scores against a warm cache instead of failing mid-report.

## Chat providers

Every provider implements the same `ChatPort`, so switching is configuration only:

```bash
CHAT__PROVIDER=openai
OPENAI__BASE_URL=http://localhost:11434/v1   # any OpenAI-compatible endpoint
OPENAI__MODEL=qwen2.5
OPENAI__API_KEY=ollama
```

`ChatSettings` (`CHAT__`) holds sampling and retry policy shared by all adapters. Provider modules only construct the LangChain model; message mapping, streaming, chaos hooks and error normalisation live in `atlas_llm/langchain_chat.py`. Embeddings and reranking stay on NVIDIA NIM (`langchain-nvidia-ai-endpoints`), so retrieval does not change with the chat provider.

## Deployment (serverless-friendly)

Two deployables, one image recipe each:

- **API** (`Dockerfile.api`): stateless FastAPI/uvicorn container → Cloud Run / Fly Machines / ECS Fargate. Scale on concurrency; `/healthz` for liveness and `/readyz` for readiness. Session state lives only in Postgres/Redis.
- **Ingestion worker** (`Dockerfile.worker`): `arq` consumer → Cloud Run Jobs + a scheduler, Fly process group, or a small always-on worker. It is horizontally scalable; jobs are retried (`max_tries=3`) and the pipeline sets `processing → indexed|failed`, so re-running an unfinished document is safe. For bursty corpora, put ingestion behind your own queue producer — `QueuePort` isolates the arq choice.
- **Migrations**: run `alembic upgrade head` as a release/deploy step (or a one-shot job), never at app startup.

Both containers run `uv sync --frozen --no-dev` and use the same lockfile. OTLP export is opt-in via `OBSERVABILITY__OTLP_ENDPOINT`.

### Production checklist

- Rotate API keys; load them from a secret manager (`SECURITY__API_KEYS`).
- Set `PII_POLICY=mask` for regulated tenants and review masked source retention (`document_contents` stores raw bytes — move to object storage if your policy forbids it).
- Keep `CHAOS__ENABLED=false` and `OBSERVABILITY__LOG_JSON=true` in prod.
- Add per-tenant budgets/alerts on whichever provider chat traffic uses; adapters retry transient errors through the LangChain client's `max_retries` (`CHAT__MAX_RETRIES`).
- Rerank: leave `NVIDIA__RERANK_BASE_URL` empty for the hosted `nemotron-3.5-rerank-8b` endpoint, or set it to a self-hosted NIM/vLLM deployment root (e.g. `http://localhost:8000/v1`); the LangChain client derives `/ranking` from that root. Validate the model name against your account/region before rollout.

## Development

```bash
make sync       # uv sync
make lint       # ruff
make typecheck  # mypy --strict
make test       # pytest
make preflight  # all three
make migrate    # alembic upgrade head
```

Tests are service-free: the pipeline test uses in-memory repositories, and the API tests never touch Postgres/Redis.

## Roadmap (deliberately out of scope)

Ragas/TruLens integration on top of the built-in evaluation harness, object storage for uploads, hierarchical summary levels (RAPTOR), per-request model budgets, JWT/OIDC auth, MMR diversity, and true late chunking once a token-level embedding provider is added. The ports make each of these additive rather than invasive.
