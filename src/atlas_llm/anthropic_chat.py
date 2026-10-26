"""Anthropic chat adapter built on `langchain-anthropic`."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from atlas_core.config import AnthropicSettings, ChatSettings
from atlas_core.security.chaos import Chaos
from atlas_llm.langchain_chat import LangChainChat


class AnthropicChat(LangChainChat):
    def __init__(
        self, settings: AnthropicSettings, chat: ChatSettings, chaos: Chaos | None = None
    ) -> None:
        model = ChatAnthropic(
            model_name=settings.model,
            api_key=settings.api_key or "not-configured",
            base_url=settings.base_url,
            temperature=chat.temperature,
            max_tokens_to_sample=chat.max_tokens,
            timeout=chat.request_timeout,
            max_retries=chat.max_retries,
            stop=None,
        )
        super().__init__(model, provider="anthropic", chaos=chaos)
