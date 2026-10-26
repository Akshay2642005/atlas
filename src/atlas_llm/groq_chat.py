"""Groq chat adapter built on `langchain-groq`."""

from __future__ import annotations

from langchain_groq import ChatGroq

from atlas_core.config import ChatSettings, GroqSettings
from atlas_core.security.chaos import Chaos
from atlas_llm.langchain_chat import LangChainChat


class GroqChat(LangChainChat):
    def __init__(
        self, settings: GroqSettings, chat: ChatSettings, chaos: Chaos | None = None
    ) -> None:
        model = ChatGroq(
            model=settings.model,
            api_key=settings.api_key or "not-configured",
            base_url=settings.base_url,
            temperature=chat.temperature,
            max_tokens=chat.max_tokens,
            timeout=chat.request_timeout,
            max_retries=chat.max_retries,
        )
        super().__init__(model, provider="groq", chaos=chaos)
