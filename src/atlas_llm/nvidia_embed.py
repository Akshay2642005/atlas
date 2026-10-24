"""NVIDIA NIM embedding adapter (OpenAI-compatible embeddings endpoint)."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import batched

from openai import APIError, AsyncOpenAI, RateLimitError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from atlas_core.config import NvidiaSettings
from atlas_core.domain.errors import InfraError
from atlas_core.domain.ports import EmbedKind
from atlas_core.security.chaos import Chaos


class NvidiaEmbedder:
    def __init__(self, settings: NvidiaSettings, chaos: Chaos | None = None) -> None:
        self._settings = settings
        self._chaos = chaos
        self._client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url, timeout=30.0)

    async def embed(self, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]:
        if self._chaos is not None:
            await self._chaos.step("nvidia.embed")
        out: list[list[float]] = []
        for batch in batched(texts, self._settings.batch_size):
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(4),
                    wait=wait_exponential_jitter(initial=1, max=8),
                    retry=retry_if_exception_type(RateLimitError),
                    reraise=True,
                ):
                    with attempt:
                        response = await self._client.embeddings.create(
                            model=self._settings.embedding_model,
                            input=list(batch),
                            extra_body={"input_type": kind.value, "truncate": self._settings.truncate},
                        )
                out.extend([d.embedding for d in response.data])
            except APIError as exc:
                raise InfraError(f"nvidia.embed failed: {exc}") from exc
        return out

    @property
    def dimensions(self) -> int:
        return self._settings.embedding_dimensions
