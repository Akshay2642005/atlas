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
    base_url: str = "https://api.groq.com/openai/v1"
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
    embedding_model: str = "nvidia/nv-embedqa-e5-v5"
    embedding_dimensions: int = 1024
    truncate: Literal["NONE", "END", "START"] = "END"
    batch_size: int = 32
    rerank_model: str = "nvidia/llama-3.2-nv-rerankqa-1b-v1"


class PostgresSettings(BaseModel):
    url: str = "postgresql+asyncpg://atlas:atlas@localhost:5432/atlas"
    pool_size: int = 10
    statement_timeout_ms: int = 8000


class RedisSettings(BaseModel):
    url: str = "redis://localhost:6379/0"


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
