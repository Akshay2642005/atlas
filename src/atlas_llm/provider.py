"""Dishka provider for LLM adapters."""

from __future__ import annotations

from dishka import Provider, Scope, provide

from atlas_core.config import AtlasSettings
from atlas_core.security import Chaos
from atlas_llm.groq_chat import GroqChat
from atlas_llm.nvidia_embed import NvidiaEmbedder
from atlas_llm.nvidia_rerank import NvidiaReranker


class LlmProvider(Provider):
    @provide(scope=Scope.APP)
    def chat(self, settings: AtlasSettings, chaos: Chaos) -> GroqChat:
        return GroqChat(settings.groq, chaos)

    @provide(scope=Scope.APP)
    def embedder(self, settings: AtlasSettings, chaos: Chaos) -> NvidiaEmbedder:
        return NvidiaEmbedder(settings.nvidia, chaos)

    @provide(scope=Scope.APP)
    def reranker(self, settings: AtlasSettings, chaos: Chaos) -> NvidiaReranker:
        return NvidiaReranker(settings.nvidia, chaos)


def providers() -> list[Provider]:
    return [LlmProvider()]
