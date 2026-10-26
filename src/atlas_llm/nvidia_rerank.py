"""NVIDIA rerank adapter built on `langchain-nvidia-ai-endpoints`."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document as LangChainDocument
from langchain_nvidia_ai_endpoints import NVIDIARerank

from atlas_core.config import NvidiaSettings
from atlas_core.domain.errors import InfraError
from atlas_core.security.chaos import Chaos
from atlas_llm.nvidia_registry import register_hosted_rerank

_INDEX_KEY = "atlas_index"
_SCORE_KEY = "relevance_score"


class NvidiaReranker:
    def __init__(self, settings: NvidiaSettings, chaos: Chaos | None = None) -> None:
        self._chaos = chaos
        client_options: dict[str, object] = {}
        if settings.rerank_base_url:
            # Self-hosted NIM/vLLM exposes the ranking route under its own root.
            client_options["base_url"] = settings.rerank_base_url
        else:
            register_hosted_rerank(settings.rerank_model)

        self._model = NVIDIARerank(
            model=settings.rerank_model,
            api_key=settings.api_key or "not-configured",
            max_batch_size=settings.batch_size,
            usage_telemetry_enabled=False,
            **client_options,
        )

    async def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        if self._chaos is not None:
            await self._chaos.step("nvidia.rerank")

        candidates = [
            LangChainDocument(page_content=text, metadata={_INDEX_KEY: index})
            for index, text in enumerate(documents)
        ]
        # Keep every candidate so scores map back onto the caller's input order.
        ranker = self._model.model_copy(update={"top_n": len(candidates)})
        try:
            ranked = await ranker.acompress_documents(candidates, query)
        except Exception as exc:  # noqa: BLE001 — normalise provider-specific errors
            raise InfraError(f"nvidia.rerank failed: {exc}") from exc

        scores = [0.0] * len(candidates)
        for document in ranked:
            index = document.metadata.get(_INDEX_KEY)
            if isinstance(index, int) and 0 <= index < len(scores):
                scores[index] = float(document.metadata.get(_SCORE_KEY, 0.0))
        return scores
