"""NVIDIA NIM embedding adapter built on `langchain-nvidia-ai-endpoints`."""

from __future__ import annotations

from collections.abc import Sequence

import structlog
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from atlas_core.config import NvidiaSettings
from atlas_core.domain.errors import InfraError
from atlas_core.domain.ports import EmbedKind
from atlas_core.security.chaos import Chaos
from atlas_llm.nvidia_registry import register_hosted_embedding

_log = structlog.get_logger(__name__)

# Retrying these cannot help: the request itself is wrong or unauthorised.
_NON_RETRYABLE = ("[400]", "[401]", "[403]", "[404]", "[410]", "[422]")


def _is_retryable(exc: Exception) -> bool:
    message = str(exc)
    return not any(marker in message for marker in _NON_RETRYABLE)


class NvidiaEmbedder:
    def __init__(self, settings: NvidiaSettings, chaos: Chaos | None = None) -> None:
        self._settings = settings
        self._chaos = chaos
        register_hosted_embedding(settings.embedding_model, settings.base_url)
        self._model = NVIDIAEmbeddings(
            model=settings.embedding_model,
            api_key=settings.api_key or "not-configured",
            base_url=settings.base_url,
            # Without this the model returns its native width, which silently
            # disobeys the vector column's declared dimension.
            dimensions=settings.embedding_dimensions,
            truncate=settings.truncate,
            max_batch_size=settings.batch_size,
            usage_telemetry_enabled=False,
        )
        # The LangChain client hard-codes a 60s aiohttp read timeout and does
        # not expose it; a cold 8b model can exceed that. Widen the read window
        # for first-use latency without touching the rest of the transport.
        self._model._async_client.timeout = settings.embedding_timeout_seconds  # pyright: ignore[reportPrivateUsage]

    async def embed(self, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]:
        if not texts:
            return []
        if self._chaos is not None:
            await self._chaos.step("nvidia.embed")
        # NIM's 8b embedder can 504 on a cold start; one retry covers that
        # without retrying auth/model errors that will never succeed.
        for attempt in (1, 2):
            try:
                return await self._embed(texts, kind)
            except Exception as exc:  # noqa: BLE001 — normalise provider-specific errors
                if attempt == 2 or not _is_retryable(exc):
                    raise InfraError(f"nvidia.embed failed: {exc}") from exc
                _log.warning("nvidia.embed.retry", error=str(exc))
        raise AssertionError("unreachable")

    async def _embed(self, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]:
        if kind is EmbedKind.QUERY:
            return [await self._model.aembed_query(text) for text in texts]
        return await self._model.aembed_documents(list(texts))

    @property
    def dimensions(self) -> int:
        return self._settings.embedding_dimensions
