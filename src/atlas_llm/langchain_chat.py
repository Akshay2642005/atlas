"""Shared LangChain chat adapter.

Translates the domain `ChatMessage` shape into LangChain messages and exposes the
LangChain model through `ChatPort`. Provider modules only build the underlying
`BaseChatModel`; retries, chaos hooks, streaming and error normalisation live
here so every provider behaves identically.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from atlas_core.domain.errors import InfraError
from atlas_core.domain.models import ChatMessage
from atlas_core.security.chaos import Chaos


def _to_langchain(messages: Sequence[ChatMessage]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for message in messages:
        if message.role == "system":
            out.append(SystemMessage(content=message.content))
        elif message.role == "assistant":
            out.append(AIMessage(content=message.content))
        else:
            out.append(HumanMessage(content=message.content))
    return out


class LangChainChat:
    """`ChatPort` implementation over any LangChain `BaseChatModel`."""

    def __init__(
        self,
        model: BaseChatModel,
        *,
        provider: str,
        token_kwarg: str = "max_tokens",
        chaos: Chaos | None = None,
    ) -> None:
        self._model = model
        self._provider = provider
        self._token_kwarg = token_kwarg
        self._chaos = chaos

    def _params(self, max_tokens: int | None) -> dict[str, Any]:
        return {} if max_tokens is None else {self._token_kwarg: max_tokens}

    async def complete(self, messages: Sequence[ChatMessage], *, max_tokens: int | None = None) -> str:
        if self._chaos is not None:
            await self._chaos.step(f"{self._provider}.chat")
        try:
            response = await self._model.ainvoke(_to_langchain(messages), **self._params(max_tokens))
        except Exception as exc:  # noqa: BLE001 — normalise provider-specific errors
            raise InfraError(f"{self._provider}.chat failed: {exc}") from exc
        return response.text

    def stream(
        self, messages: Sequence[ChatMessage], *, max_tokens: int | None = None
    ) -> AsyncIterator[str]:
        return self._stream(messages, max_tokens=max_tokens)

    async def _stream(
        self, messages: Sequence[ChatMessage], *, max_tokens: int | None = None
    ) -> AsyncIterator[str]:
        if self._chaos is not None:
            await self._chaos.step(f"{self._provider}.chat.stream")
        try:
            async for chunk in self._model.astream(
                _to_langchain(messages), **self._params(max_tokens)
            ):
                if text := chunk.text:
                    yield text
        except Exception as exc:  # noqa: BLE001 — normalise provider-specific errors
            raise InfraError(f"{self._provider}.chat.stream failed: {exc}") from exc
