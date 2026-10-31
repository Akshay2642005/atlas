"""Typed settings. Nested groups via `<GROUP>__<FIELD>` env vars (see .env.example)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChatSettings(BaseModel):
    """Provider-independent chat behaviour. `provider` selects the adapter."""

    provider: Literal["groq", "openai", "anthropic", "nvidia"] = "groq"
    temperature: float = 0.2
    max_tokens: int = 1024
    request_timeout: float = 60.0
    max_retries: int = 2


class GroqSettings(BaseModel):
    api_key: str = ""
    # Unset by default: `langchain-groq` already knows the Groq endpoint, so a
    # configured root only helps when routing through a proxy. Note the SDK
    # appends /openai/v1 itself, so do not include that path here.
    base_url: str | None = None
    model: str = "llama-3.3-70b-versatile"


class OpenAISettings(BaseModel):
    """OpenAI or any OpenAI-compatible endpoint (set `base_url`)."""

    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"


class AnthropicSettings(BaseModel):
    api_key: str = ""
    base_url: str = "https://api.anthropic.com"
    model: str = "claude-haiku-4-5-20251001"


class NvidiaSettings(BaseModel):
    api_key: str = ""
    base_url: str = "https://integrate.api.nvidia.com/v1"
    chat_model: str = "meta/llama-3.1-8b-instruct"
    # 4096-native: `nv-embedqa-e5-v5` is EOL (410 Gone since 2026-08-25) and
    # `dimensions` can only truncate, never widen, a model's output.
    embedding_model: str = "nvidia/nemotron-3.5-embed-8b"
    embedding_dimensions: int = 4096
    # The 8b embedder can take >60s on a cold NIM start, which is the client's
    # built-in read timeout. Give the first request enough headroom to finish.
    embedding_timeout_seconds: float = 180.0
    truncate: Literal["NONE", "END", "START"] = "END"
    batch_size: int = 32
    # Hosted Nemotron reranking is served from the retrieval API, not the
    # OpenAI-compatible `/v1` root used for chat and embeddings.
    rerank_model: str = "nvidia/nemotron-3.5-rerank-8b"
    # Empty selects the hosted endpoint registered for `rerank_model` above.
    # Set this to a NIM deployment root (e.g. `http://localhost:8000/v1`) to
    # rerank against self-hosted NIM/vLLM instead.
    rerank_base_url: str = ""


class PostgresSettings(BaseModel):
    url: str = "postgresql+asyncpg://atlas:atlas@localhost:5432/atlas"
    pool_size: int = 10
    statement_timeout_ms: int = 8000


class RedisSettings(BaseModel):
    url: str = "redis://localhost:6379/0"


class StorageSettings(BaseModel):
    """Object storage for document bytes.

    One adapter covers S3-compatible stores (AWS S3, RustFS, MinIO), GCS, and a
    local directory. `local` is the default so a fresh clone runs without any
    infrastructure; it is not meant for deployment.
    """

    backend: Literal["local", "s3", "gcs"] = "local"
    bucket: str = "atlas"
    # Applied inside the bucket, so several deployments can share one bucket.
    prefix: str = ""
    # RustFS/MinIO: set the endpoint and keep path-style addressing. Empty means
    # AWS S3, where obstore derives the endpoint from the region.
    endpoint: str | None = None
    region: str = "us-east-1"
    access_key_id: str = ""
    secret_access_key: str = ""
    force_path_style: bool = False
    allow_http: bool = False
    local_root: str = ".atlas-storage"
    # Objects above this size are spilled to a temp file instead of memory when
    # a loader needs the whole payload (PDF text extraction does).
    max_inline_bytes: int = 100 * 1024 * 1024


class ApiSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class ChunkingSettings(BaseModel):
    chunk_size_tokens: int = 400
    overlap_tokens: int = 60
    enable_semantic_chunks: bool = False
    semantic_percentile: float = 0.80


class RetrievalSettings(BaseModel):
    semantic_top_k: int = 50
    lexical_top_k: int = 50
    # Reranking cost scales with candidate count, so cap the fused list before
    # sending it to the model. 24 is the smallest depth measured to rank
    # identically to 50 on the evaluation corpus, at half the latency.
    rerank_top_k: int = 24
    final_top_k: int = 6
    enable_rerank: bool = True
    rrf_k: int = 60


class AnswerSettings(BaseModel):
    system_prompt: str = (
        "You are Atlas, an enterprise RAG assistant. Answer ONLY from the numbered context "
        "passages. Cite every claim inline as [n] where n is the passage number. "
        "If the context is insufficient, say you don't know."
    )
    max_context_chunks: int = 8


class SecuritySettings(BaseModel):
    api_keys: str = ""  # "<key>=<tenant_id>,<key>=<tenant_id>"
    pii_policy: Literal["off", "mask"] = "mask"
    disable_injection_guard: bool = False
    rate_limit_rpm: int = 120

    def api_key_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for pair in self.api_keys.split(","):
            if "=" in (pair := pair.strip()):
                key, tenant = pair.split("=", 1)
                if key and tenant:
                    out[key] = tenant
        return out


class ObservabilitySettings(BaseModel):
    log_level: str = "INFO"
    log_json: bool = False
    service_name: str = "atlas"
    otlp_endpoint: str = ""


class ChaosSettings(BaseModel):
    enabled: bool = False
    failure_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    sleep_seconds: float = 0.0


class AtlasSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__", extra="ignore")

    env: Literal["local", "dev", "staging", "prod"] = "local"
    app_name: str = "atlas"

    chat: ChatSettings = Field(default_factory=ChatSettings)
    groq: GroqSettings = Field(default_factory=GroqSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    nvidia: NvidiaSettings = Field(default_factory=NvidiaSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    answer: AnswerSettings = Field(default_factory=AnswerSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    chaos: ChaosSettings = Field(default_factory=ChaosSettings)


@lru_cache(maxsize=1)
def load_settings() -> AtlasSettings:
    return AtlasSettings()  # type: ignore[call-arg]
