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

All provider adapters use LangChain integration packages behind `ChatPort` / `EmbedderPort` / `RerankerPort`:

- **Groq:** `langchain-groq` (`ChatGroq`) — fast Llama inference; default chat provider
- **OpenAI:** `langchain-openai` (`ChatOpenAI`) — official API plus any OpenAI-compatible `base_url`
- **Anthropic:** `langchain-anthropic` (`ChatAnthropic`) — Claude models
- **NVIDIA:** `langchain-nvidia-ai-endpoints` (`ChatNVIDIA`, `NVIDIAEmbeddings`, `NVIDIARerank`) — NIM chat, embeddings, reranking
- **Provider selection:** `CHAT__PROVIDER` (`groq|openai|anthropic|nvidia`) resolved by `atlas_llm/provider.py:build_chat`
- **Google Gemini:** not implemented; would be another `ChatPort` adapter (`langchain-google-genai`)

## Infrastructure

- **Containerization:** Docker
- **Orchestration:** Kubernetes (via Helm)
- **Reverse Proxy:** Nginx
- **Observability:** Jaeger (tracing), Prometheus (metrics)

## Frontend (TBD)

- Framework: To be decided (likely Next.js based on product spec)
