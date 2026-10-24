# Atlas Technology Stack

## Backend

- **Language:** Python 3.12
- **Framework:** FastAPI
- **ORM:** SQLAlchemy (async)
- **Database:** PostgreSQL 15+ with pgVector extension
- **Cache/Queue:** Redis 7+
- **DI Framework:** Dishka
- **Validation:** Pydantic
- **Configuration:** Pydantic Settings
- **Logging:** Structlog
- **Tracing:** OpenTelemetry
- **Job Queue:** Arq
- **Testing:** pytest, pytest-asyncio
- **Linting:** ruff
- **Type Checking:** mypy, basedpyright

## LLM & Embedding Providers

- **Groq:** Fast LLM inference (Llama models)
- **NVIDIA:** Embeddings and reranking
- **Anthropic:** High-quality LLM (Claude models)
- **OpenAI:** Widely-used LLM (GPT models)
- **Google Gemini:** Multimodal LLM

## Infrastructure

- **Containerization:** Docker
- **Orchestration:** Kubernetes (via Helm)
- **Reverse Proxy:** Nginx
- **Observability:** Jaeger (tracing), Prometheus (metrics)

## Frontend (TBD)

- Framework: To be decided (likely Next.js based on product spec)
