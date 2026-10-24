# Atlas Design Plan

## Architecture Overview

Atlas follows hexagonal architecture with ports and adapters:

### Core (atlas_core)
- Domain models (entities)
- Ports (Protocol interfaces)
- Configuration
- Security (PII, injection guard, chaos)
- Observability (logging, tracing)

### Adapters
- **atlas_llm:** LLM providers (Groq, Anthropic, OpenAI, Google, NVIDIA)
- **atlas_vector:** PostgreSQL with pgVector
- **atlas_ingest:** Document ingestion pipeline
- **atlas_retrieval:** Retrieval and answer generation
- **atlas_api:** FastAPI REST API
- **atlas_worker:** Background job processing

### DI Container

Dishka container wires all providers:
- Core providers (settings, ports)
- LLM providers
- Vector providers
- Ingest providers
- Retrieval providers
- API providers

## Multi-Tenancy

Tenant isolation at every layer:
- Database: tenant_id column in all tables
- Queries: WHERE tenant_id = ?
- API: API key validation per tenant
- Rate limiting: Per-tenant token bucket

## Scalability

- Connection pooling (SQLAlchemy)
- Partition pruning (pgVector by tenant_id)
- Async processing (Redis Streams)
- Response caching (Redis)
- Advanced indexing (HNSW, IVFFlat, DiskANN)
