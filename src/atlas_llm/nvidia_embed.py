"""NVIDIA NIM embedding adapter built on `langchain-nvidia-ai-endpoints`."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from atlas_core.config import NvidiaSettings
from atlas_core.domain.errors import InfraError
from atlas_core.domain.ports import EmbedKind
from atlas_core.security.chaos import Chaos


class NvidiaEmbedder:
    def __init__(self, settings: NvidiaSettings, chaos: Chaos | None = None) -> None:
        self._settings = settings
        self._chaos = chaos
        self._model = NVIDIAEmbeddings(
            model=settings.embedding_model,
            api_key=settings.api_key or "not-configured",
            base_url=settings.base_url,
            truncate=settings.truncate,
            max_batch_size=settings.batch_size,
            usage_telemetry_enabled=False,
        )

    async def embed(self, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]:
        if not texts:
            return []
        if self._chaos is not None:
            await self._chaos.step("nvidia.embed")
        try:
            if kind is EmbedKind.QUERY:
                return [await self._model.aembed_query(text) for text in texts]
            return await self._model.aembed_documents(list(texts))
        except Exception as exc:  # noqa: BLE001 — normalise provider-specific errors
            raise InfraError(f"nvidia.embed failed: {exc}") from exc

    @property
    def dimensions(self) -> int:
        return self._settings.embedding_dimensions
