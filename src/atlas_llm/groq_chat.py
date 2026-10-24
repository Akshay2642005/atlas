"""Groq chat adapter (OpenAI-compatible endpoint)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from openai import APIError, AsyncOpenAI, RateLimitError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from atlas_core.config import GroqSettings
from atlas_core.domain.errors import InfraError
from atlas_core.domain.models import ChatMessage
from atlas_core.security.chaos import Chaos


def _payload(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in messages]


class GroqChat:
    def __init__(self, settings: GroqSettings, chaos: Chaos | None = None) -> None:
        self._settings = settings
        self._chaos = chaos
        self._client = AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.request_timeout,
        )

    async def complete(self, messages: Sequence[ChatMessage], *, max_tokens: int | None = None) -> str:
        if self._chaos is not None:
            await self._chaos.step("groq.chat")
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(4),
                wait=wait_exponential_jitter(initial=1, max=8),
                retry=retry_if_exception_type(RateLimitError),
                reraise=True,
            ):
                with attempt:
                    response = await self._client.chat.completions.create(
                        model=self._settings.model,
                        messages=_payload(messages),  # type: ignore[arg-type]
                        temperature=self._settings.temperature,
                        max_tokens=max_tokens or self._settings.max_tokens,
                    )
            return response.choices[0].message.content or ""
        except APIError as exc:
            raise InfraError(f"groq.chat failed: {exc}") from exc

    async def stream(self, messages: Sequence[ChatMessage], *, max_tokens: int | None = None) -> AsyncIterator[str]:
        if self._chaos is not None:
            await self._chaos.step("groq.chat.stream")
        try:
            stream = await self._client.chat.completions.create(
                model=self._settings.model,
                messages=_payload(messages),  # type: ignore[arg-type]
                temperature=self._settings.temperature,
                max_tokens=max_tokens or self._settings.max_tokens,
                stream=True,
            )
        except APIError as exc:
            raise InfraError(f"groq.chat.stream failed: {exc}") from exc
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
