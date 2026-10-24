# Atlas Enterprise RAG Platform - Complete Product Specification

**Version:** 1.0  
**Date:** 2025-01-18  
**Status:** Draft  
**Target Release:** Week 8 (8-week development cycle)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Product Positioning](#product-positioning)
3. [Technical Architecture](#technical-architecture)
4. [Feature Specifications](#feature-specifications)
   - [Web Dashboard](#1-web-dashboard)
   - [Multi-Provider LLM Support](#2-multi-provider-llm-support)
   - [Advanced pgVector Indexing](#3-advanced-pgvector-indexing)
   - [Tenant-Based Partitioning](#4-tenant-based-partitioning)
   - [Hybrid Search (BM25 + Vector)](#5-hybrid-search-bm25--vector)
   - [NVIDIA Reranking](#6-nvidia-reranking)
   - [Presidio PII Detection](#7-presidio-pii-detection)
   - [Redis Streams Async Ingestion](#8-redis-streams-async-ingestion)
   - [Document Versioning](#9-document-versioning)
   - [Token Bucket Rate Limiting](#10-token-bucket-rate-limiting)
   - [Connection Pooling](#11-connection-pooling)
   - [Response Caching](#12-response-caching)
   - [Enhanced Observability](#13-enhanced-observability)
   - [Chunking Strategy Configurations](#14-chunking-strategy-configurations)
5. [Data Model](#data-model)
6. [API Specification](#api-specification)
7. [Configuration](#configuration)
8. [Deployment](#deployment)
9. [Security & Compliance](#security--compliance)
10. [Implementation Roadmap](#implementation-roadmap)

---

## Executive Summary

Atlas is a self-hosted, web-based RAG platform that gives teams complete control over their retrieval-augmented generation stack. Deploy it once on your own infrastructure, then manage everything through a single web dashboard.

**Key Differentiators:**
- Self-hosted SaaS - Your data, your infra, our UI
- Multi-provider LLM support - Groq, Anthropic, OpenAI, Google (switch via env keys)
- Model selection UI - Choose models per tenant from the web dashboard
- Zero-code operation - Everything in the browser (documents, queries, settings)
- Type-safe internals - Refactor confidence for engineering teams

**Target Release:** Week 8 (8-week development cycle)

---

## Product Positioning

### Target Customers

- **Mid-market companies** (50-500 employees) with data sovereignty requirements
- **Enterprise teams** locked into strict compliance (HIPAA, SOC 2, GDPR)
- **Engineering teams** who distrust "black box" SaaS RAG solutions
- **Organizations** with multi-tenant requirements (serve multiple customers from one deployment)

### Competitive Positioning

| Feature | Atlas | Pinecone | LangChain Cloud | Haystack Platform |
|---------|-------|----------|----------------|------------------|
| Self-hosted | ✅ | ❌ | ❌ | Partial |
| Multi-provider LLM | ✅ | ❌ | ✅ | ❌ |
| Web Dashboard | ✅ | ✅ | ✅ | ✅ |
| Multi-tenancy | ✅ | ✅ | ✅ | ✅ |
| Type-safe internals | ✅ | ❌ | ❌ | ❌ |
| Cost efficiency | 10x cheaper | Expensive | Expensive | Expensive |
| PII Detection | ✅ | ❌ | ✅ | ✅ |
| Observability | ✅ | ✅ | ✅ | ✅ |

### Pricing Model

**Self-hosted license pricing:**
- **Starter**: $299/month (up to 5 tenants, 10K documents)
- **Professional**: $999/month (up to 20 tenants, 100K documents)
- **Enterprise**: $2,999/month (unlimited tenants, unlimited documents)

**What customers get:**
- Docker image of Atlas (web dashboard + backend)
- License key to activate
- Documentation and deployment guides
- Email support
- Updates for 1 year

**Customer pays additionally:**
- Their own infra costs (PostgreSQL, Redis, GPU if needed)

---

## Technical Architecture

### System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Dashboard (Next.js/React)           │
│  - Document upload, query playground, tenant management     │
│  - Settings, observability, embedded Jaeger/Grafana         │
└────────────────────┬──────────────────────────────────────┘
                     │ REST API
         ┌───────────┴───────────┐
         │   FastAPI Backend    │  (Atlas Core)
         │  - DI container      │
         │  - Multi-provider LLM │
         │  - RAG pipeline      │
         └───────────┬───────────┘
                     │
    ┌────────────────┼────────────────┐
    │                │                │
┌───▼────┐     ┌────▼────┐     ┌────▼────┐
│  Redis │     │Postgres │     │  Redis  │
│ (Cache)│     │(Vector  │     │(Queue)  │
│        │     │ + Data) │     │         │
└────────┘     └────┬────┘     └─────────┘
                    │
        ┌───────────┼───────────┐
        │  pgVector  │  Partitions│
        │  (HNSW/    │  (tenant_id)│
        │  IVFFlat)  │            │
        └───────────┴───────────┘
```

### Tech Stack

**Frontend (New):**
- **Next.js 14** - React framework with SSR/SSG
- **shadcn/ui** - Component library
- **Tailwind CSS** - Styling
- **TanStack Query** - API client
- **Zustand** - State management
- **Recharts** - Charts for metrics dashboard

**Backend (Existing + New):**
- **FastAPI** - REST API
- **Atlas Core** - Existing RAG pipeline
- **Dishka** - DI container
- **SQLAlchemy** - ORM
- **pgVector** - Vector database extension
- **Alembic** - Database migrations

**Infrastructure:**
- **PostgreSQL 15+** - Data + vector storage
- **Redis 7+** - Cache + job queue
- **NVIDIA API** - Embeddings + reranking
- **Groq/Anthropic/OpenAI/Google** - LLM providers

---

## Feature Specifications

### 1. Web Dashboard

#### 1.1 Document Management

**UI Components:**
- **Upload Zone**: Drag-and-drop area for files
- **File List**: Table showing documents with columns:
  - Document name
  - Tenant
  - Status (pending, processing, indexed, failed)
  - Chunk count
  - Created at
  - Actions (delete, view, retry)
- **Progress Bar**: Real-time ingestion progress per document
- **Metadata Form**: Custom key-value pairs per document
- **Tenant Selector**: Dropdown to assign tenant

**Features:**
- Upload single or multiple files
- Supported formats: PDF, Word, Excel, PowerPoint, Text, Markdown (100+ formats via pypdf)
- Batch upload with progress tracking
- Real-time status updates (via WebSocket or polling)
- Delete documents with confirmation
- Filter documents by tenant, status, metadata
- View document versions and rollback

**API Endpoints:**
```
POST   /api/documents/upload           # Upload document
GET    /api/documents                  # List documents
GET    /api/documents/{id}             # Get document details
DELETE /api/documents/{id}             # Delete document
GET    /api/documents/{id}/versions    # Get version history
POST   /api/documents/{id}/rollback    # Rollback to version
GET    /api/documents/{id}/status     # Get ingestion status
```

#### 1.2 Query Playground

**UI Components:**
- **Chat Interface**: Message history with user/assistant bubbles
- **Input Field**: Text area for questions
- **Citations Display**: Inline citations `[1]`, `[2]` in responses
- **Chunk Inspector**: Side panel showing retrieved chunks with:
  - Chunk text
  - Score
  - Source document
  - Chunk number
- **Model Selector**: Dropdown to choose LLM model for testing
- **Settings Panel**: Adjust top-k, blend ratio, reranking threshold
- **Saved Searches**: List of saved queries for reuse

**Features:**
- Streaming responses (real-time token streaming)
- Citations with clickable links to source chunks
- Compare model outputs (switch models instantly)
- Save queries as "saved searches"
- Export query results (JSON, CSV)
- Toggle between semantic/hybrid search
- Adjust retrieval parameters with sliders

**API Endpoints:**
```
POST   /api/query/stream               # Streaming query
POST   /api/query                     # Non-streaming query
GET    /api/query/{id}                # Get query history
POST   /api/query/save                # Save query
GET    /api/query/saved               # List saved queries
```

#### 1.3 Tenant Management

**UI Components:**
- **Tenant List**: Table showing tenants with columns:
  - Tenant name
  - API key (masked)
  - Document count
  - Query count
  - Status (active, suspended)
  - Actions (configure, delete)
- **Tenant Settings Form**: Per-tenant configuration:
  - LLM provider selection
  - Model selection
  - Chunking strategy
  - Hybrid search blend ratio
  - Rate limits
  - System prompt
- **API Key Management**: Generate, rotate, revoke API keys

**Features:**
- Create new tenants with API key generation
- Configure per-tenant settings
- View tenant metrics (queries, documents, errors)
- Manage API keys (rotate, revoke)
- Set rate limits per tenant
- Suspend/activate tenants

**API Endpoints:**
```
POST   /api/tenants                   # Create tenant
GET    /api/tenants                   # List tenants
GET    /api/tenants/{id}              # Get tenant details
PUT    /api/tenants/{id}              # Update tenant settings
DELETE /api/tenants/{id}              # Delete tenant
POST   /api/tenants/{id}/api-key     # Rotate API key
GET    /api/tenants/{id}/metrics     # Get tenant metrics
```

#### 1.4 Settings & Configuration

**UI Components:**
- **Provider Configuration**: Input fields for API keys:
  - Groq API key
  - Anthropic API key
  - OpenAI API key
  - Google API key
- **Model Selection**: Dropdowns per provider showing available models
- **Index Configuration**: 
  - Index type (HNSW, IVFFlat)
  - Partitioning toggle
  - Quantization settings
- **Observability Settings**:
  - OTLP endpoint
  - Log level
  - Trace sampling rate
- **System Settings**:
  - Default chunk size
  - Default overlap
  - Max context chunks

**Features:**
- Auto-detect providers from env keys
- Test provider connectivity
- Configure model selection per tenant
- Adjust indexing parameters
- Configure observability backends
- View system health status

**API Endpoints:**
```
GET    /api/providers                 # List available providers
GET    /api/providers/{id}/models    # List models for provider
PUT    /api/settings                 # Update system settings
GET    /api/settings                 # Get system settings
GET    /api/health                    # System health check
```

#### 1.5 Observability Dashboard

**UI Components:**
- **Metrics Dashboard**: Charts showing:
  - Queries per second (per tenant)
  - Average latency (p50, p95, p99)
  - Error rate (per tenant, per provider)
  - Cache hit rate
  - Index size growth
  - Token usage per model
- **Trace Explorer**: Embedded Jaeger/Zipkin UI
- **Log Viewer**: Real-time log stream with filters
- **Health Status**: Green/red indicators for system components

**Features:**
- Real-time metrics updates
- Filter by tenant, provider, time range
- Click any query to view full trace
- View span timings, error rates, token counts
- Export metrics as CSV

**API Endpoints:**
```
GET    /api/metrics                   # Get system metrics
GET    /api/traces/{id}              # Get trace details
GET    /api/logs                      # Get logs (with filters)
```

---

### 2. Multi-Provider LLM Support

#### 2.1 Supported Providers

**Groq**
- API Key: `GROQ_API_KEY`
- Models: llama-3.3-70b-versatile, llama-3.1-70b-versatile, mixtral-8x7b-32768
- Strength: Fast inference, low cost

**Anthropic**
- API Key: `ANTHROPIC_API_KEY`
- Models: claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022
- Strength: High-quality responses, long context

**OpenAI**
- API Key: `OPENAI_API_KEY`
- Models: gpt-4o, gpt-4o-mini, gpt-4-turbo
- Strength: Widely-used, good performance

**Google Gemini**
- API Key: `GOOGLE_API_KEY`
- Models: gemini-1.5-pro, gemini-1.5-flash
- Strength: Multimodal, long context

#### 2.2 Provider Detection Logic

**Implementation:**
```python
# atlas_llm/provider.py
class LlmProviderSelector:
    def __init__(self, settings: AtlasSettings):
        self._providers = []
        
        if settings.groq.api_key:
            self._providers.append(("groq", GroqChat(settings.groq)))
        
        if settings.anthropic.api_key:
            self._providers.append(("anthropic", AnthropicChat(settings.anthropic)))
        
        if settings.openai.api_key:
            self._providers.append(("openai", OpenAIChat(settings.openai)))
        
        if settings.google.api_key:
            self._providers.append(("google", GoogleChat(settings.google)))
    
    def get_provider(self, provider_name: str) -> ChatPort:
        for name, provider in self._providers:
            if name == provider_name:
                return provider
        raise ValueError(f"Provider {provider_name} not configured")
    
    def available_providers(self) -> list[str]:
        return [name for name, _ in self._providers]
```

**Model Registry:**
```python
MODEL_REGISTRY = {
    "groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "mixtral-8x7b-32768",
    ],
    "anthropic": [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
    ],
    "google": [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ],
}
```

#### 2.3 Per-Tenant Model Selection

**Configuration:**
```python
# atlas_core/config.py
class TenantSettings(BaseModel):
    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    embedding_provider: str = "nvidia"
    embedding_model: str = "nvidia/nv-embedqa-e5-v5"
    rerank_provider: str = "nvidia"
    rerank_model: str = "nvidia/nv-rerankqa-mistral-4b-v3"
    chunk_size_tokens: int = 400
    overlap_tokens: int = 60
    enable_semantic_chunks: bool = False
    semantic_percentile: float = 0.80
    semantic_top_k: int = 50
    lexical_top_k: int = 50
    final_top_k: int = 6
    enable_rerank: bool = True
    rrf_k: int = 60
    hybrid_alpha: float = 0.5  # 0.0 = BM25 only, 1.0 = vector only
    max_context_chunks: int = 8
    system_prompt: str = (
        "You are Atlas, an enterprise RAG assistant. Answer ONLY from the numbered context "
        "passages. Cite every claim inline as [n] where n is the passage number. "
        "If the context is insufficient, say you don't know."
    )
    rate_limit_rpm: int = 120
```

**DI Integration:**
```python
# atlas_retrieval/provider.py
class RetrievalProvider(Provider):
    @provide
    def answer_service(
        self,
        settings: AtlasSettings,
        chat: ChatPort,
        retriever: Retriever,
        pii: PiiDetectorPort,
        guard: HeuristicInjectionGuard,
        tenant_settings: TenantSettings,  # Loaded from DB per request
    ) -> AnswerService:
        # Select LLM provider based on tenant settings
        selector = container.get(LlmProviderSelector)
        llm = selector.get_provider(tenant_settings.llm_provider)
        
        return AnswerService(
            settings=settings.answer,
            chat=llm,
            retriever=retriever,
            pii=pii,
            guard=guard,
            tenant_settings=tenant_settings,
        )
```

#### 2.4 Provider Fallback

**Configuration:**
```python
class ProviderFallback(BaseModel):
    primary_provider: str = "groq"
    fallback_providers: list[str] = ["anthropic", "openai"]
    max_retries: int = 3
    retry_delay_ms: int = 1000
```

**Implementation:**
```python
class ResilientChatPort(ChatPort):
    def __init__(self, selector: LlmProviderSelector, fallback: ProviderFallback):
        self._selector = selector
        self._fallback = fallback
    
    async def complete(self, messages: Sequence[ChatMessage], *, max_tokens: int | None = None) -> str:
        providers = [self._fallback.primary_provider] + self._fallback.fallback_providers
        
        for provider_name in providers:
            try:
                provider = self._selector.get_provider(provider_name)
                return await provider.complete(messages, max_tokens=max_tokens)
            except InfraError as exc:
                if provider_name == providers[-1]:
                    raise
                await asyncio.sleep(self._fallback.retry_delay_ms / 1000)
        
        raise InfraError("All providers failed")
```

---

### 3. Advanced pgVector Indexing

#### 3.1 Index Types

**HNSW (Hierarchical Navigable Small World)**
- Best for: Fast approximate search
- Pros: Sub-millisecond queries, good for high-dimensional vectors
- Cons: Higher memory usage, slower index build
- Configuration:
  ```sql
  CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
  ```

**IVFFlat (Inverted File with Flat)**
- Best for: High-accuracy search
- Pros: Lower memory usage, faster index build
- Cons: Slower queries, requires tuning `lists` parameter
- Configuration:
  ```sql
  CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);
  ```

**DiskANN (Disk-based Approximate Nearest Neighbor)**
- Best for: Large-scale datasets (100M+ vectors)
- Pros: Low RAM usage, scales to disk
- Cons: Requires pgvectorscale extension
- Configuration:
  ```sql
  CREATE INDEX ON chunks USING diskann (embedding)
  WITH (num_neighbors = 50, max_alpha = 1.0);
  ```

#### 3.2 Index Configuration

**Schema Extension:**
```python
# atlas_vector/schema.py
class IndexSettings(BaseModel):
    index_type: Literal["hnsw", "ivfflat", "diskann"] = "hnsw"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    hnsw_ef_search: int = 40
    ivfflat_lists: int = 100
    diskann_num_neighbors: int = 50
    diskann_max_alpha: float = 1.0
```

**Migration Script:**
```python
# alembic/versions/001_advanced_indexing.py
def upgrade():
    # HNSW index
    op.execute("""
        CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
        ON chunks USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)
    
    # IVFFlat index (optional, can be created manually)
    op.execute("""
        COMMENT ON INDEX chunks_embedding_hnsw IS
        'HNSW index for vector similarity search';
    """)

def downgrade():
    op.execute("DROP INDEX IF EXISTS chunks_embedding_hnsw;")
```

#### 3.3 Quantization Support

**Binary Quantization**
- Reduces memory by 32x (1024-dim float → 1024 bits)
- Minimal accuracy loss for large datasets
- Configuration:
  ```sql
  ALTER TABLE chunks ALTER COLUMN embedding SET DATA TYPE bit(1024);
  ```

**Scalar Quantization**
- Reduces memory by 4x (float32 → int8)
- Better accuracy than binary, higher memory usage
- Configuration:
  ```sql
  ALTER TABLE chunks ALTER COLUMN embedding SET DATA TYPE smallint[];
  ```

**Implementation:**
```python
class QuantizationSettings(BaseModel):
    enabled: bool = False
    method: Literal["binary", "scalar"] = "scalar"
    compression_ratio: int = 4  # 4 for scalar, 32 for binary
```

---

### 4. Tenant-Based Partitioning

#### 4.1 Partitioning Strategy

**Partition by tenant_id:**
```sql
-- Create partitioned table
CREATE TABLE chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL,
    tenant_id TEXT NOT NULL,
    seq INT NOT NULL,
    text TEXT NOT NULL,
    embedding vector(1024),
    created_at TIMESTAMP DEFAULT NOW()
) PARTITION BY LIST (tenant_id);

-- Create partitions for known tenants
CREATE TABLE chunks_tenant_a PARTITION OF chunks FOR VALUES IN ('tenant_a');
CREATE TABLE chunks_tenant_b PARTITION OF chunks FOR VALUES IN ('tenant_b');

-- Default partition for new tenants
CREATE TABLE chunks_default PARTITION OF chunks DEFAULT;
```

**Automatic partition creation:**
```python
# atlas_vector/partitions.py
class PartitionManager:
    async def ensure_partition(self, tenant_id: str):
        """Create partition for tenant if it doesn't exist."""
        partition_name = f"chunks_{tenant_id.replace('-', '_')}"
        
        async with self._session() as session:
            exists = await session.execute(
                text("""
                    SELECT EXISTS (
                        SELECT FROM pg_tables
                        WHERE schemaname = 'public'
                        AND tablename = :partition_name
                    )
                """),
                {"partition_name": partition_name}
            )
            
            if not exists.scalar():
                await session.execute(
                    text(f"""
                        CREATE TABLE {partition_name} PARTITION OF chunks
                        FOR VALUES IN ('{tenant_id}')
                    """)
                )
                await session.commit()
```

#### 4.2 Partition Pruning

**Query optimization:**
```sql
-- Query with tenant_id filter automatically prunes partitions
EXPLAIN ANALYZE
SELECT * FROM chunks
WHERE tenant_id = 'tenant_a'
AND embedding <-> '[0.1, 0.2, ...]' LIMIT 10;

-- Output shows: "Partition Pruning: chunks_tenant_a"
```

**Performance impact:**
- 10x faster queries for multi-tenant workloads
- Reduced I/O (only scans relevant partition)
- Better cache locality

---

### 5. Hybrid Search (BM25 + Vector)

#### 5.1 BM25 Implementation

**TSVECTOR column:**
```sql
-- Add TSVECTOR column to chunks table
ALTER TABLE chunks ADD COLUMN tsv tsvector
GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;

-- Create GIN index
CREATE INDEX chunks_tsv_gin ON chunks USING GIN (tsv);
```

**BM25 retriever:**
```python
# atlas_retrieval/bm25_retriever.py
class Bm25Retriever:
    async def search(
        self,
        tenant_id: str,
        query: str,
        *,
        k: int = 50,
    ) -> list[tuple[Chunk, float]]:
        async with self._session_factory() as session:
            result = await session.execute(
                text("""
                    SELECT id, document_id, tenant_id, seq, text,
                           ts_rank(tsv, plainto_tsquery('english', :query)) as score
                    FROM chunks
                    WHERE tenant_id = :tenant_id
                    AND tsv @@ plainto_tsquery('english', :query)
                    ORDER BY score DESC
                    LIMIT :k
                """),
                {"tenant_id": tenant_id, "query": query, "k": k}
            )
            
            rows = result.fetchall()
            return [(self._row_to_chunk(row), row.score) for row in rows]
```

#### 5.2 Reciprocal Rank Fusion (RRF)

**RRF algorithm:**
```python
def reciprocal_rank_fusion(
    semantic_results: list[tuple[Chunk, float]],
    bm25_results: list[tuple[Chunk, float]],
    k: int = 60,
) -> list[tuple[Chunk, float]]:
    """Fuse semantic and BM25 results using RRF."""
    scores: dict[uuid.UUID, float] = {}
    
    # Score semantic results
    for rank, (chunk, _) in enumerate(semantic_results, 1):
        scores[chunk.id] = scores.get(chunk.id, 0) + 1.0 / (k + rank)
    
    # Score BM25 results
    for rank, (chunk, _) in enumerate(bm25_results, 1):
        scores[chunk.id] = scores.get(chunk.id, 0) + 1.0 / (k + rank)
    
    # Sort by fused score
    chunk_map = {chunk.id: chunk for chunk, _ in semantic_results + bm25_results}
    sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return [(chunk_map[chunk_id], score) for chunk_id, score in sorted_chunks]
```

#### 5.3 Hybrid Search with Alpha Blend

**Alpha-blended fusion:**
```python
def alpha_blend_fusion(
    semantic_results: list[tuple[Chunk, float]],
    bm25_results: list[tuple[Chunk, float]],
    alpha: float = 0.5,  # 0.0 = BM25 only, 1.0 = vector only
) -> list[tuple[Chunk, float]]:
    """Fuse results using alpha-blended score."""
    scores: dict[uuid.UUID, float] = {}
    
    # Normalize semantic scores to [0, 1]
    if semantic_results:
        max_semantic = max(score for _, score in semantic_results)
        for chunk, score in semantic_results:
            normalized = score / max_semantic if max_semantic > 0 else 0
            scores[chunk.id] = scores.get(chunk.id, 0) + alpha * normalized
    
    # Normalize BM25 scores to [0, 1]
    if bm25_results:
        max_bm25 = max(score for _, score in bm25_results)
        for chunk, score in bm25_results:
            normalized = score / max_bm25 if max_bm25 > 0 else 0
            scores[chunk.id] = scores.get(chunk.id, 0) + (1 - alpha) * normalized
    
    # Sort by fused score
    chunk_map = {chunk.id: chunk for chunk, _ in semantic_results + bm25_results}
    sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return [(chunk_map[chunk_id], score) for chunk_id, score in sorted_chunks]
```

**Configuration:**
```python
class HybridSearchSettings(BaseModel):
    method: Literal["rrf", "alpha_blend"] = "alpha_blend"
    alpha: float = 0.5  # For alpha_blend
    rrf_k: int = 60  # For RRF
    semantic_top_k: int = 50
    bm25_top_k: int = 50
    final_top_k: int = 6
```

---

### 6. NVIDIA Reranking

#### 6.1 Reranking Pipeline

**Two-stage retrieval:**
```python
# atlas_retrieval/reranking_retriever.py
class RerankingRetriever:
    def __init__(
        self,
        base_retriever: Retriever,
        reranker: RerankerPort,
        top_k: int = 50,
        final_k: int = 6,
    ):
        self._base_retriever = base_retriever
        self._reranker = reranker
        self._top_k = top_k
        self._final_k = final_k
    
    async def search(
        self,
        tenant_id: str,
        query: str,
        embedding: Sequence[float],
    ) -> list[tuple[Chunk, float]]:
        # Stage 1: Retrieve top-k chunks
        candidates = await self._base_retriever.search(
            tenant_id, embedding, k=self._top_k
        )
        
        # Stage 2: Rerank with NVIDIA
        candidate_texts = [chunk.text for chunk, _ in candidates]
        rerank_scores = await self._reranker.rerank(query, candidate_texts)
        
        # Combine chunks with rerank scores
        reranked = list(zip([chunk for chunk, _ in candidates], rerank_scores))
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked[:self._final_k]
```

#### 6.2 NVIDIA Reranker Configuration

**Settings:**
```python
class NvidiaRerankSettings(BaseModel):
    api_key: str = ""
    base_url: str = "https://ai.api.nvidia.com/v1/reranking"
    model: str = "nvidia/nv-rerankqa-mistral-4b-v3"
    top_n: int = 6
    batch_size: int = 32
```

**Implementation:**
```python
# atlas_llm/nvidia_rerank.py
class NvidiaReranker:
    async def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        payload = {
            "model": self._settings.model,
            "query": query,
            "documents": list(documents),
            "top_n": self._settings.top_n,
        }
        
        response = await self._client.post(
            self._settings.base_url,
            json=payload,
            headers={"Authorization": f"Bearer {self._settings.api_key}"},
        )
        
        result = response.json()
        return [item["logprob"] for item in result["results"]]
```

---

### 7. Presidio PII Detection

#### 7.1 Presidio Integration

**Installation:**
```bash
uv add presidio spacy
python -m spacy download en_core_web_lg
```

**PII Detector Implementation:**
```python
# atlas_core/security/presidio_pii.py
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

class PresidioPiiDetector:
    def __init__(self):
        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()
    
    def detect(self, text: str) -> list[PiiFinding]:
        """Detect PII entities in text."""
        results = self._analyzer.analyze(
            text=text,
            language="en",
            entities=[
                "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
                "IBAN_CODE", "CREDIT_CARD", "IP_ADDRESS",
                "LOCATION", "DATE_TIME", "URL", "US_SSN",
            ],
        )
        
        return [
            PiiFinding(
                kind=result.entity_type,
                start=result.start,
                end=result.end,
                value=text[result.start:result.end],
            )
            for result in results
        ]
    
    def anonymize(self, text: str) -> str:
        """Anonymize PII in text."""
        results = self._analyzer.analyze(text=text, language="en")
        anonymized = self._anonymizer.anonymize(
            text=text,
            analyzer_results=results,
        )
        return anonymized.text
```

#### 7.2 DI Integration

**Provider:**
```python
# atlas_core/security/provider.py
class SecurityProvider(Provider):
    @provide
    def pii_detector(self, settings: AtlasSettings) -> PiiDetectorPort:
        if settings.security.pii_policy == "presidio":
            return PresidioPiiDetector()
        else:
            return RegexPiiDetector()  # Fallback to regex
```

#### 7.3 PII Detection in Ingestion Pipeline

**Integration:**
```python
# atlas_ingest/pipeline.py
class IngestionPipeline:
    async def ingest(self, document: Document, content: bytes) -> None:
        # Parse document
        text = await self._loader.load(content)
        
        # Detect PII
        pii_findings = self._pii_detector.detect(text)
        
        # Anonymize if policy is "mask"
        if self._settings.security.pii_policy == "mask":
            text = self._pii_detector.anonymize(text)
        
        # Chunk and embed
        chunks = await self._chunker.chunk(text)
        embeddings = await self._embedder.embed([c.text for c in chunks], kind=EmbedKind.PASSAGE)
        
        # Store
        await self._vector_repo.upsert_chunks(document.tenant_id, document.id, zip(chunks, embeddings))
```

---

### 8. Redis Streams Async Ingestion

#### 8.1 Redis Streams Architecture

**Stream structure:**
```python
# Stream: "ingest:jobs"
# Fields:
#   - tenant_id: str
#   - document_id: UUID
#   - status: "pending" | "processing" | "completed" | "failed"
#   - error: str (optional)
```

**Producer (API):**
```python
# atlas_ingest/queue.py
class RedisQueueProducer:
    async def enqueue_ingest(self, tenant_id: str, document_id: uuid.UUID) -> str:
        job_id = str(uuid.uuid4())
        
        await self._redis.xadd(
            "ingest:jobs",
            {
                "job_id": job_id,
                "tenant_id": tenant_id,
                "document_id": str(document_id),
                "status": "pending",
                "timestamp": str(time.time()),
            },
        )
        
        return job_id
```

**Consumer (Worker):**
```python
# atlas_worker/processor.py
class IngestionWorker:
    async def process_jobs(self):
        consumer_group = "ingest:workers"
        consumer_name = f"worker-{socket.gethostname()}"
        
        # Create consumer group if it doesn't exist
        try:
            await self._redis.xgroup_create("ingest:jobs", consumer_group, id="0", mkstream=True)
        except ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
        
        while True:
            # Read from stream
            messages = await self._redis.xreadgroup(
                consumer_group,
                consumer_name,
                {"ingest:jobs": ">"},
                count=1,
                block=5000,  # 5 second timeout
            )
            
            for stream, entries in messages:
                for message_id, fields in entries:
                    await self._process_message(message_id, fields)
```

#### 8.2 Task Metadata Tracking

**Task table:**
```sql
CREATE TABLE ingestion_tasks (
    id UUID PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    document_id UUID NOT NULL,
    status TEXT NOT NULL,  -- pending, processing, completed, failed
    error TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    retry_count INT DEFAULT 0,
    max_retries INT DEFAULT 3
);
```

**Task tracking:**
```python
class TaskTracker:
    async def create_task(self, tenant_id: str, document_id: uuid.UUID) -> uuid.UUID:
        task_id = uuid.uuid4()
        async with self._session_factory() as session:
            await session.execute(
                insert(IngestionTask).values(
                    id=task_id,
                    tenant_id=tenant_id,
                    document_id=document_id,
                    status="pending",
                )
            )
            await session.commit()
        return task_id
    
    async def update_status(
        self,
        task_id: uuid.UUID,
        status: str,
        error: str | None = None,
    ):
        async with self._session_factory() as session:
            await session.execute(
                update(IngestionTask)
                .where(IngestionTask.id == task_id)
                .values(status=status, error=error, updated_at=datetime.now())
            )
            await session.commit()
```

---

### 9. Document Versioning

#### 9.1 Versioning Schema

**Version table:**
```sql
CREATE TABLE document_versions (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL,
    tenant_id TEXT NOT NULL,
    version_number INT NOT NULL,
    content_hash TEXT NOT NULL,
    chunk_count INT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    created_by TEXT,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX document_versions_unique ON document_versions (document_id, version_number);
```

**File-to-document link:**
```sql
CREATE TABLE file_documents (
    id UUID PRIMARY KEY,
    file_uri TEXT NOT NULL,
    document_id UUID NOT NULL,
    tenant_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);
```

#### 9.2 Version Management

**Create version:**
```python
class DocumentVersioner:
    async def create_version(
        self,
        document_id: uuid.UUID,
        tenant_id: str,
        content: bytes,
        chunk_count: int,
        created_by: str,
    ) -> int:
        # Calculate content hash
        content_hash = hashlib.sha256(content).hexdigest()
        
        # Get current version number
        async with self._session_factory() as session:
            result = await session.execute(
                select(func.max(DocumentVersion.version_number))
                .where(DocumentVersion.document_id == document_id)
            )
            current_version = result.scalar() or 0
            new_version = current_version + 1
            
            # Create version record
            await session.execute(
                insert(DocumentVersion).values(
                    id=uuid.uuid4(),
                    document_id=document_id,
                    tenant_id=tenant_id,
                    version_number=new_version,
                    content_hash=content_hash,
                    chunk_count=chunk_count,
                    created_by=created_by,
                )
            )
            await session.commit()
            
            return new_version
```

**Rollback to version:**
```python
async def rollback_to_version(
    self,
    document_id: uuid.UUID,
    version_number: int,
    tenant_id: str,
):
    # Get version details
    async with self._session_factory() as session:
        result = await session.execute(
            select(DocumentVersion)
            .where(
                DocumentVersion.document_id == document_id,
                DocumentVersion.version_number == version_number,
            )
        )
        version = result.scalar_one()
        
        # Delete current chunks
        await session.execute(
            delete(ChunkRow)
            .where(ChunkRow.document_id == document_id)
        )
        
        # Restore chunks from version (assuming we store chunk snapshots)
        # This requires additional table design for chunk snapshots
        await session.commit()
```

---

### 10. Token Bucket Rate Limiting

#### 10.1 Token Bucket Algorithm

**Implementation:**
```python
# atlas_core/rate_limit.py
class TokenBucketRateLimiter:
    def __init__(self, redis: Redis, rate: int, period: int = 60):
        self._redis = redis
        self._rate = rate  # tokens per period
        self._period = period  # seconds
    
    async def check_rate_limit(self, tenant_id: str) -> bool:
        """Check if tenant is within rate limit."""
        key = f"rate_limit:{tenant_id}"
        
        # Get current bucket state
        state = await self._redis.hgetall(key)
        
        if not state:
            # First request, initialize bucket
            await self._redis.hset(key, mapping={
                "tokens": self._rate - 1,
                "last_refill": time.time(),
            })
            await self._redis.expire(key, self._period)
            return True
        
        tokens = int(state[b"tokens"])
        last_refill = float(state[b"last_refill"])
        
        # Refill tokens based on elapsed time
        now = time.time()
        elapsed = now - last_refill
        refill_amount = int(elapsed * self._rate / self._period)
        tokens = min(self._rate, tokens + refill_amount)
        
        if tokens > 0:
            # Consume one token
            await self._redis.hset(key, mapping={
                "tokens": tokens - 1,
                "last_refill": now,
            })
            return True
        else:
            # Rate limit exceeded
            return False
```

#### 10.2 Per-Tenant Configuration

**Configuration:**
```python
class RateLimitSettings(BaseModel):
    enabled: bool = True
    default_rpm: int = 120  # requests per minute
    tenant_limits: dict[str, int] = {}  # tenant_id -> rpm override
```

**DI Integration:**
```python
class RateLimitProvider(Provider):
    @provide
    def rate_limiter(
        self,
        redis: Redis,
        settings: RateLimitSettings,
    ) -> TokenBucketRateLimiter:
        return TokenBucketRateLimiter(
            redis=redis,
            rate=settings.default_rpm,
            period=60,
        )
```

**Middleware:**
```python
# atlas_api/middlewares.py
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    tenant_id = request.headers.get("X-Tenant-ID")
    
    if tenant_id:
        limiter = container.get(TokenBucketRateLimiter)
        allowed = await limiter.check_rate_limit(tenant_id)
        
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded"},
            )
    
    return await call_next(request)
```

---

### 11. Connection Pooling

#### 11.1 SQLAlchemy Connection Pool

**Configuration:**
```python
# atlas_vector/engine.py
def create_engine_and_session(
    settings: PostgresSettings,
    *,
    instrument: bool = False,
) -> tuple[AsyncEngine, SessionFactory]:
    engine = create_async_engine(
        settings.url,
        pool_size=settings.pool_size,
        max_overflow=10,  # Additional connections when pool is exhausted
        pool_pre_ping=True,  # Verify connections before use
        pool_recycle=3600,  # Recycle connections after 1 hour
        connect_args={
            "server_settings": {
                "statement_timeout": str(settings.statement_timeout_ms)
            }
        },
    )
    
    if instrument:
        SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine)
    
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    
    return engine, session_factory
```

#### 11.2 Pool Monitoring

**Metrics:**
```python
class PoolMetrics:
    def __init__(self, engine: AsyncEngine):
        self._engine = engine
    
    def get_pool_status(self) -> dict:
        pool = self._engine.pool
        return {
            "pool_size": pool.size(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "checked_in": pool.checkedin(),
        }
```

---

### 12. Response Caching

#### 12.1 Redis Cache Implementation

**Cache key:**
```python
def cache_key(tenant_id: str, query: str, model: str) -> str:
    """Generate cache key for query."""
    key_data = f"{tenant_id}:{model}:{query}"
    return f"cache:{hashlib.sha256(key_data.encode()).hexdigest()}"
```

**Cache decorator:**
```python
# atlas_core/cache.py
class ResponseCache:
    def __init__(self, redis: Redis, ttl: int = 3600):
        self._redis = redis
        self._ttl = ttl
    
    async def get(self, key: str) -> str | None:
        cached = await self._redis.get(key)
        return cached.decode() if cached else None
    
    async def set(self, key: str, value: str):
        await self._redis.setex(key, self._ttl, value)
    
    async def invalidate_tenant(self, tenant_id: str):
        """Invalidate all cache entries for a tenant."""
        pattern = f"cache:*:{tenant_id}:*"
        keys = await self._redis.keys(pattern)
        if keys:
            await self._redis.delete(*keys)
```

**Integration:**
```python
class CachedAnswerService(AnswerService):
    def __init__(self, base_service: AnswerService, cache: ResponseCache):
        self._base = base_service
        self._cache = cache
    
    async def answer(self, query: Query) -> Answer:
        # Check cache
        key = cache_key(query.tenant_id, query.text, self._model)
        cached = await self._cache.get(key)
        
        if cached:
            return Answer.model_validate_json(cached)
        
        # Generate answer
        answer = await self._base.answer(query)
        
        # Cache result
        await self._cache.set(key, answer.model_dump_json())
        
        return answer
```

---

### 13. Enhanced Observability

#### 13.1 OpenTelemetry Fanout

**Multiple backends:**
```python
# atlas_core/observability.py
def setup_tracing(settings: ObservabilitySettings) -> TracerProvider | None:
    if not settings.otlp_endpoint:
        return None
    
    provider = TracerProvider(resource=Resource(attributes={"service.name": settings.service_name}))
    
    # Fanout to multiple backends
    backends = settings.otlp_endpoint.split(",")  # "endpoint1,endpoint2"
    
    for endpoint in backends:
        try:
            exporter = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint.strip()))
            provider.add_span_processor(exporter)
        except Exception:
            exporter = BatchSpanProcessor(ConsoleSpanExporter())
            provider.add_span_processor(exporter)
    
    trace.set_tracer_provider(provider)
    return provider
```

#### 13.2 Trace Attributes

**Span attributes:**
```python
# atlas_retrieval/answerer.py
@tracer.start_as_current_span("answer_service.answer")
async def answer(self, query: Query) -> Answer:
    span = trace.get_current_span()
    span.set_attribute("tenant_id", query.tenant_id)
    span.set_attribute("query_length", len(query.text))
    span.set_attribute("model", self._tenant_settings.llm_model)
    
    # Retrieval span
    with tracer.start_as_current_span("retrieval"):
        chunks = await self._retriever.search(query.tenant_id, query.text)
        span.set_attribute("retrieved_chunks", len(chunks))
    
    # Generation span
    with tracer.start_as_current_span("generation"):
        answer = await self._chat.complete(messages)
        span.set_attribute("answer_length", len(answer))
    
    return answer
```

#### 13.3 Metrics Export

**Prometheus metrics:**
```python
# atlas_core/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
query_counter = Counter(
    "atlas_queries_total",
    "Total number of queries",
    ["tenant_id", "model", "status"],
)

query_latency = Histogram(
    "atlas_query_latency_seconds",
    "Query latency in seconds",
    ["tenant_id", "model"],
)

chunks_retrieved = Gauge(
    "atlas_chunks_retrieved",
    "Number of chunks retrieved",
    ["tenant_id"],
)

# Use in code
@query_latency.labels(tenant_id=tenant_id, model=model).time()
async def answer(query: Query) -> Answer:
    chunks_retrieved.labels(tenant_id=tenant_id).set(len(chunks))
    query_counter.labels(tenant_id=tenant_id, model=model, status="success").inc()
    return answer
```

---

### 14. Chunking Strategy Configurations

#### 14.1 Chunking Strategies

**Fixed-size chunking:**
```python
class FixedSizeChunker:
    def __init__(self, chunk_size: int, overlap: int):
        self._chunk_size = chunk_size
        self._overlap = overlap
    
    async def chunk(self, text: str) -> list[ChunkDraft]:
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self._chunk_size
            chunk_text = text[start:end]
            chunks.append(ChunkDraft(text=chunk_text, seq=len(chunks)))
            start = end - self._overlap
        
        return chunks
```

**Semantic chunking:**
```python
class SemanticChunker:
    def __init__(self, percentile: float = 0.8):
        self._percentile = percentile
    
    async def chunk(self, text: str) -> list[ChunkDraft]:
        # Split by sentences
        sentences = text.split(". ")
        
        # Embed sentences
        embeddings = await self._embedder.embed(sentences, kind=EmbedKind.PASSAGE)
        
        # Calculate similarity scores
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
        
        # Find breakpoint at percentile
        threshold = np.percentile(similarities, self._percentile * 100)
        
        # Chunk at breakpoints
        chunks = []
        current_chunk = []
        
        for i, (sentence, sim) in enumerate(zip(sentences, similarities + [0])):
            current_chunk.append(sentence)
            
            if sim < threshold:
                chunks.append(ChunkDraft(text=". ".join(current_chunk), seq=len(chunks)))
                current_chunk = []
        
        if current_chunk:
            chunks.append(ChunkDraft(text=". ".join(current_chunk), seq=len(chunks)))
        
        return chunks
```

#### 14.2 Per-Document Type Configuration

**Configuration:**
```python
class ChunkingStrategyConfig(BaseModel):
    default: ChunkingSettings = Field(default_factory=ChunkingSettings)
    by_mime_type: dict[str, ChunkingSettings] = {
        "application/pdf": ChunkingSettings(chunk_size_tokens=500, overlap_tokens=100),
        "text/markdown": ChunkingSettings(chunk_size_tokens=300, overlap_tokens=50),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ChunkingSettings(
            chunk_size_tokens=400, overlap_tokens=60
        ),
    }
```

---

## Data Model

### Core Tables

**documents**
```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    mime TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    checksum TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    error TEXT
);

CREATE INDEX documents_tenant_idx ON documents(tenant_id);
CREATE INDEX documents_status_idx ON documents(status);
```

**chunks**
```sql
CREATE TABLE chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL,
    tenant_id TEXT NOT NULL,
    seq INT NOT NULL,
    level INT DEFAULT 0,
    text TEXT NOT NULL,
    token_count INT DEFAULT 0,
    embedding vector(1024),
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX chunks_document_idx ON chunks(document_id);
CREATE INDEX chunks_tenant_idx ON chunks(tenant_id);
CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX chunks_tsv_gin ON chunks USING GIN (tsv);
```

**document_content**
```sql
CREATE TABLE document_content (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL,
    content BYTEA NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);
```

**tenants**
```sql
CREATE TABLE tenants (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    api_key TEXT UNIQUE NOT NULL,
    settings JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    status TEXT DEFAULT 'active'
);
```

**tenant_settings**
```sql
CREATE TABLE tenant_settings (
    tenant_id UUID PRIMARY KEY REFERENCES tenants(id) ON DELETE CASCADE,
    llm_provider TEXT DEFAULT 'groq',
    llm_model TEXT DEFAULT 'llama-3.3-70b-versatile',
    embedding_provider TEXT DEFAULT 'nvidia',
    embedding_model TEXT DEFAULT 'nvidia/nv-embedqa-e5-v5',
    chunk_size_tokens INT DEFAULT 400,
    overlap_tokens INT DEFAULT 60,
    enable_semantic_chunks BOOLEAN DEFAULT FALSE,
    semantic_percentile FLOAT DEFAULT 0.80,
    semantic_top_k INT DEFAULT 50,
    lexical_top_k INT DEFAULT 50,
    final_top_k INT DEFAULT 6,
    enable_rerank BOOLEAN DEFAULT TRUE,
    rrf_k INT DEFAULT 60,
    hybrid_alpha FLOAT DEFAULT 0.5,
    max_context_chunks INT DEFAULT 8,
    system_prompt TEXT,
    rate_limit_rpm INT DEFAULT 120
);
```

**document_versions**
```sql
CREATE TABLE document_versions (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL,
    tenant_id TEXT NOT NULL,
    version_number INT NOT NULL,
    content_hash TEXT NOT NULL,
    chunk_count INT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    created_by TEXT,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX document_versions_unique ON document_versions (document_id, version_number);
```

**ingestion_tasks**
```sql
CREATE TABLE ingestion_tasks (
    id UUID PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    document_id UUID NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    retry_count INT DEFAULT 0,
    max_retries INT DEFAULT 3
);
```

---

## API Specification

### Authentication

**API Key Authentication:**
```http
GET /api/documents
Authorization: Bearer <tenant-api-key>
X-Tenant-ID: <tenant-id>
```

### Endpoints

#### Documents

**Upload Document**
```http
POST /api/documents/upload
Content-Type: multipart/form-data

{
  "file": <binary>,
  "tenant_id": "tenant-123",
  "metadata": {
    "category": "legal",
    "department": "hr"
  }
}

Response: 202 Accepted
{
  "document_id": "uuid",
  "status": "pending",
  "job_id": "uuid"
}
```

**List Documents**
```http
GET /api/documents?tenant_id=tenant-123&status=indexed&limit=50

Response: 200 OK
{
  "documents": [
    {
      "id": "uuid",
      "tenant_id": "tenant-123",
      "source_uri": "contract.pdf",
      "mime": "application/pdf",
      "status": "indexed",
      "chunk_count": 150,
      "created_at": "2025-01-18T10:00:00Z"
    }
  ],
  "total": 100
}
```

**Get Document Status**
```http
GET /api/documents/{id}/status

Response: 200 OK
{
  "document_id": "uuid",
  "status": "processing",
  "progress": 0.6,
  "chunk_count": 90,
  "total_chunks": 150
}
```

#### Query

**Streaming Query**
```http
POST /api/query/stream
Content-Type: application/json

{
  "tenant_id": "tenant-123",
  "query": "What is the notice period?",
  "model": "claude-3-5-sonnet-20241022",
  "citations": true,
  "top_k": 6
}

Response: 200 OK (Server-Sent Events)
data: {"token": "The"}
data: {"token": " notice"}
data: {"token": " period"}
...
data: {"done": true, "citations": [...]}
```

**Non-Streaming Query**
```http
POST /api/query
Content-Type: application/json

{
  "tenant_id": "tenant-123",
  "query": "What is the notice period?",
  "model": "claude-3-5-sonnet-20241022",
  "citations": true,
  "top_k": 6
}

Response: 200 OK
{
  "answer": "The notice period is 30 days [1], but may be reduced to 14 days [2]...",
  "citations": [
    {
      "document_id": "uuid",
      "chunk_id": "uuid",
      "source_uri": "contract.pdf",
      "seq": 45,
      "score": 0.89,
      "text": "The notice period for termination is 30 days..."
    }
  ],
  "trace_id": "uuid"
}
```

#### Tenants

**Create Tenant**
```http
POST /api/tenants
Content-Type: application/json

{
  "name": "Acme Corp",
  "settings": {
    "llm_provider": "anthropic",
    "llm_model": "claude-3-5-sonnet-20241022",
    "chunk_size_tokens": 500
  }
}

Response: 201 Created
{
  "tenant_id": "uuid",
  "name": "Acme Corp",
  "api_key": "atlas_tenant_...",
  "settings": {...}
}
```

**Update Tenant Settings**
```http
PUT /api/tenants/{id}
Content-Type: application/json

{
  "llm_provider": "groq",
  "llm_model": "llama-3.3-70b-versatile",
  "hybrid_alpha": 0.7
}

Response: 200 OK
```

#### Providers

**List Available Providers**
```http
GET /api/providers

Response: 200 OK
{
  "providers": [
    {
      "name": "groq",
      "models": ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile"],
      "configured": true
    },
    {
      "name": "anthropic",
      "models": ["claude-3-5-sonnet-20241022"],
      "configured": true
    }
  ]
}
```

#### Metrics

**Get System Metrics**
```http
GET /api/metrics?tenant_id=tenant-123&from=2025-01-01&to=2025-01-18

Response: 200 OK
{
  "queries_per_second": 15.5,
  "average_latency_p50": 0.85,
  "average_latency_p95": 1.2,
  "average_latency_p99": 2.1,
  "error_rate": 0.02,
  "cache_hit_rate": 0.35,
  "token_usage": {
    "groq": 150000,
    "anthropic": 50000
  }
}
```

---

## Configuration

### Environment Variables

```bash
# Environment
ATLAS_ENV=production
ATLAS_APP_NAME=atlas

# LLM Providers
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_BASE_URL=https://api.groq.com/openai/v1

ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_BASE_URL=https://api.anthropic.com

OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
OPENAI_BASE_URL=https://api.openai.com/v1

GOOGLE_API_KEY=...
GOOGLE_MODEL=gemini-1.5-pro
GOOGLE_BASE_URL=https://generativelanguage.googleapis.com/v1beta

# NVIDIA (Embeddings + Reranking)
NVIDIA_API_KEY=nvapi-...
NVIDIA_EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5
NVIDIA_RERANK_MODEL=nvidia/nv-rerankqa-mistral-4b-v3
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1

# PostgreSQL
POSTGRES_URL=postgresql+asyncpg://atlas:atlas@localhost:5432/atlas
POSTGRES_POOL_SIZE=10
POSTGRES_STATEMENT_TIMEOUT_MS=8000

# Redis
REDIS_URL=redis://localhost:6379/0

# API
API_HOST=0.0.0.0
API_PORT=8000

# Chunking
CHUNKING_CHUNK_SIZE_TOKENS=400
CHUNKING_OVERLAP_TOKENS=60
CHUNKING_ENABLE_SEMANTIC_CHUNKS=false
CHUNKING_SEMANTIC_PERCENTILE=0.80

# Retrieval
RETRIEVAL_SEMANTIC_TOP_K=50
RETRIEVAL_LEXICAL_TOP_K=50
RETRIEVAL_FINAL_TOP_K=6
RETRIEVAL_ENABLE_RERANK=true
RETRIEVAL_RRF_K=60

# Answer
ANSWER_SYSTEM_PROMPT="You are Atlas..."
ANSWER_MAX_CONTEXT_CHUNKS=8

# Security
SECURITY_API_KEYS=key1=tenant1,key2=tenant2
SECURITY_PII_POLICY=mask
SECURITY_DISABLE_INJECTION_GUARD=false
SECURITY_RATE_LIMIT_RPM=120

# Observability
OBSERVABILITY_LOG_LEVEL=INFO
OBSERVABILITY_LOG_JSON=false
OBSERVABILITY_SERVICE_NAME=atlas
OBSERVABILITY_OTLP_ENDPOINT=http://jaeger:4317

# Chaos Testing
CHAOS_ENABLED=false
CHAOS_FAILURE_RATE=0.0
CHAOS_SLEEP_SECONDS=0.0
```

### Config File (Optional)

```yaml
# atlas.yaml
app:
  name: atlas
  env: production

providers:
  groq:
    api_key: ${GROQ_API_KEY}
    model: llama-3.3-70b-versatile
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    model: claude-3-5-sonnet-20241022

indexing:
  type: hnsw
  hnsw_m: 16
  hnsw_ef_construction: 64
  partitioning: true

cache:
  enabled: true
  ttl: 3600
  backend: redis

observability:
  otlp_endpoints:
    - http://jaeger:4317
    - http://tempo:4317
  trace_sampling: 0.1
```

---

## Deployment

### Docker Compose

```yaml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_DB: atlas
      POSTGRES_USER: atlas
      POSTGRES_PASSWORD: atlas
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  atlas-api:
    build: .
    environment:
      - POSTGRES_URL=postgresql+asyncpg://atlas:atlas@postgres:5432/atlas
      - REDIS_URL=redis://redis:6379/0
      - GROQ_API_KEY=${GROQ_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis

  atlas-worker:
    build: .
    command: python -m atlas_worker.main
    environment:
      - POSTGRES_URL=postgresql+asyncpg://atlas:atlas@postgres:5432/atlas
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
      - "4317:4317"

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes Helm Chart

```yaml
# Chart.yaml
apiVersion: v2
name: atlas
description: Atlas Enterprise RAG Platform
type: application
version: 1.0.0
appVersion: "1.0.0"
```

```yaml
# values.yaml
replicaCount: 3

image:
  repository: your-registry/atlas
  tag: "1.0.0"
  pullPolicy: Always

postgres:
  enabled: true
  host: postgres-service
  database: atlas
  username: atlas
  password: atlas

redis:
  enabled: true
  host: redis-service
  port: 6379

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

---

## Security & Compliance

### PII Detection

**Supported Entity Types:**
- PERSON (names)
- EMAIL_ADDRESS
- PHONE_NUMBER
- IBAN_CODE
- CREDIT_CARD
- IP_ADDRESS
- LOCATION
- DATE_TIME
- URL
- US_SSN
- UK_NINO
- PASSPORT

**PII Policies:**
- `off`: No detection
- `mask`: Replace with placeholders (e.g., `[NAME]`, `[EMAIL]`)
- `block`: Reject documents with PII
- `anonymize`: Replace with synthetic data

### Tenant Isolation

**Row-Level Security:**
```sql
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON documents
    FOR ALL
    TO atlas_user
    USING (tenant_id = current_setting('app.tenant_id'));

ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON chunks
    FOR ALL
    TO atlas_user
    USING (tenant_id = current_setting('app.tenant_id'));
```

**Application-level isolation:**
- All queries include `WHERE tenant_id = ?`
- Partition pruning for performance
- API key validation on every request

### Audit Logging

**Log format:**
```json
{
  "timestamp": "2025-01-18T10:00:00Z",
  "tenant_id": "tenant-123",
  "user_id": "user-456",
  "operation": "document.upload",
  "resource_id": "uuid",
  "ip_address": "192.168.1.1",
  "user_agent": "Mozilla/5.0...",
  "trace_id": "uuid"
}
```

### Encryption

**At rest:**
- PostgreSQL: AES-256 encryption (managed by cloud provider or LUKS)
- Redis: Optional TLS encryption

**In transit:**
- TLS 1.3 for all API endpoints
- TLS for PostgreSQL connections
- TLS for Redis connections

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Week 1: Web Dashboard Scaffold**
- [ ] Set up Next.js project with shadcn/ui
- [ ] Create layout with navigation
- [ ] Implement document upload UI
- [ ] Implement query playground UI
- [ ] Basic tenant management UI

**Week 2: API Endpoints**
- [ ] Document upload endpoint
- [ ] Document list endpoint
- [ ] Query endpoint (streaming)
- [ ] Tenant CRUD endpoints
- [ ] Authentication middleware

### Phase 2: Multi-Provider (Weeks 3-4)

**Week 3: Provider Implementation**
- [ ] Implement Anthropic provider
- [ ] Implement OpenAI provider
- [ ] Implement Google Gemini provider
- [ ] Provider selector with DI
- [ ] Model registry

**Week 4: Model Selection UI**
- [ ] Provider auto-detection
- [ ] Model selection dropdown
- [ ] Per-tenant model configuration
- [ ] Provider fallback logic
- [ ] API endpoints for providers

### Phase 3: Enterprise Features (Weeks 5-6)

**Week 5: Indexing & Partitioning**
- [ ] Advanced pgVector indexing (HNSW/IVFFlat)
- [ ] Tenant partitioning
- [ ] Partition manager
- [ ] Index configuration UI
- [ ] Migration scripts

**Week 6: Hybrid Search & Reranking**
- [ ] BM25 retriever implementation
- [ ] TSVECTOR index
- [ ] RRF and alpha-blend fusion
- [ ] NVIDIA reranking integration
- [ ] Hybrid search configuration UI

### Phase 4: Scale & Polish (Weeks 7-8)

**Week 7: Async & Versioning**
- [ ] Redis Streams queue
- [ ] Ingestion worker
- [ ] Task tracking
- [ ] Document versioning
- [ ] Version rollback UI

**Week 8: Security & Observability**
- [ ] Presidio PII detection
- [ ] Token bucket rate limiting
- [ ] Response caching
- [ ] Enhanced observability (fanout)
- [ ] Metrics dashboard
- [ ] Documentation and deployment guides

---

## Success Criteria

### Week 8 Deliverables

**Functional:**
- ✅ Web dashboard with document upload, query playground, tenant management
- ✅ Multi-provider LLM support (Groq, Anthropic, OpenAI, Google)
- ✅ Model selection UI per tenant
- ✅ Advanced pgVector indexing (HNSW/IVFFlat)
- ✅ Tenant partitioning
- ✅ Hybrid search (BM25 + vector)
- ✅ NVIDIA reranking
- ✅ Presidio PII detection
- ✅ Redis Streams async ingestion
- ✅ Document versioning
- ✅ Token bucket rate limiting
- ✅ Response caching
- ✅ Enhanced observability (Jaeger, Prometheus)
- ✅ Configurable chunking strategies

**Non-Functional:**
- ✅ Sub-100ms query latency (p95)
- ✅ 10M+ chunks per tenant (via partitioning)
- ✅ 99.9% uptime
- ✅ SOC 2 ready (PII detection, audit logging, encryption)
- ✅ Docker Compose deployment (one-command setup)
- ✅ Helm chart for Kubernetes
- ✅ Complete documentation

**Business:**
- ✅ Demo environment
- ✅ Pricing calculator
- ✅ Feature comparison matrix
- ✅ Security whitepaper
- ✅ Sales deck

---

## Conclusion

This specification defines a complete, production-ready, self-hosted RAG platform that can be deployed in 8 weeks. The platform leverages Atlas's existing type-safe, DI-first architecture while adding enterprise-grade features through a web-based interface.

The implementation roadmap is divided into 4 phases, each building on the previous one. By Week 8, you'll have a viable commercial product competing against LangChain Cloud and Haystack Platform—differentiated by being self-hosted, multi-provider by default, and cost-efficient.
