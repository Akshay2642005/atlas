"""Dishka provider for LLM adapters (registered under their ports).

Provider packages are imported lazily: startup only loads the integration that
`CHAT__PROVIDER` actually selects, and NVIDIA adapters load on first use.
"""

from __future__ import annotations

from dishka import Provider, Scope, provide

from atlas_core.config import AtlasSettings
from atlas_core.domain.ports import ChatPort, EmbedderPort, RerankerPort
from atlas_core.security import Chaos


def build_chat(settings: AtlasSettings, chaos: Chaos | None = None) -> ChatPort:
    """Select the chat adapter named by `CHAT__PROVIDER`."""
    provider = settings.chat.provider
    if provider == "groq":
        from atlas_llm.groq_chat import GroqChat

        return GroqChat(settings.groq, settings.chat, chaos)
    if provider == "openai":
        from atlas_llm.openai_chat import OpenAIChat

        return OpenAIChat(settings.openai, settings.chat, chaos)
    if provider == "anthropic":
        from atlas_llm.anthropic_chat import AnthropicChat

        return AnthropicChat(settings.anthropic, settings.chat, chaos)
    if provider == "nvidia":
        from atlas_llm.nvidia_chat import NvidiaChat

        return NvidiaChat(settings.nvidia, settings.chat, chaos)
    raise ValueError(f"unsupported chat provider: {provider}")


class LlmProvider(Provider):
    @provide(scope=Scope.APP)
    def chat(self, settings: AtlasSettings, chaos: Chaos) -> ChatPort:
        return build_chat(settings, chaos)

    @provide(scope=Scope.APP)
    def embedder(self, settings: AtlasSettings, chaos: Chaos) -> EmbedderPort:
        from atlas_llm.nvidia_embed import NvidiaEmbedder

        return NvidiaEmbedder(settings.nvidia, chaos)

    @provide(scope=Scope.APP)
    def reranker(self, settings: AtlasSettings, chaos: Chaos) -> RerankerPort:
        from atlas_llm.nvidia_rerank import NvidiaReranker

        return NvidiaReranker(settings.nvidia, chaos)


def providers() -> list[Provider]:
    return [LlmProvider()]
