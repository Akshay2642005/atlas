"""OpenAI chat adapter built on `langchain-openai`.

Works with the official API and any OpenAI-compatible endpoint: point
`OPENAI__BASE_URL` at the provider's `/v1` root.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from atlas_core.config import ChatSettings, OpenAISettings
from atlas_core.security.chaos import Chaos
from atlas_llm.langchain_chat import LangChainChat


class OpenAIChat(LangChainChat):
    def __init__(
        self, settings: OpenAISettings, chat: ChatSettings, chaos: Chaos | None = None
    ) -> None:
        model = ChatOpenAI(
            model=settings.model,
            api_key=settings.api_key or "not-configured",  # pyright: ignore[reportArgumentType]
            base_url=settings.base_url,
            temperature=chat.temperature,
            max_completion_tokens=chat.max_tokens,
            timeout=chat.request_timeout,
            max_retries=chat.max_retries,
        )
        super().__init__(
            model,
            provider="openai",
            token_kwarg="max_completion_tokens",
            chaos=chaos,
        )
