"""NVIDIA NIM chat adapter built on `langchain-nvidia-ai-endpoints`."""

from __future__ import annotations

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from atlas_core.config import ChatSettings, NvidiaSettings
from atlas_core.security.chaos import Chaos
from atlas_llm.langchain_chat import LangChainChat


class NvidiaChat(LangChainChat):
    def __init__(
        self, settings: NvidiaSettings, chat: ChatSettings, chaos: Chaos | None = None
    ) -> None:
        model = ChatNVIDIA(
            model=settings.chat_model,
            api_key=settings.api_key or "not-configured",
            base_url=settings.base_url,
            temperature=chat.temperature,
            max_completion_tokens=chat.max_tokens,
            timeout=chat.request_timeout,
            usage_telemetry_enabled=False,
        )
        super().__init__(model, provider="nvidia", token_kwarg="max_tokens", chaos=chaos)
