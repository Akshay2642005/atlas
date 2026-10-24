"""NVIDIA rerank adapter (NIM /reranking endpoint, non-OpenAI protocol)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from atlas_core.config import NvidiaSettings
from atlas_core.domain.errors import InfraError
from atlas_core.security.chaos import Chaos


class NvidiaReranker:
    def __init__(self, settings: NvidiaSettings, chaos: Chaos | None = None) -> None:
        self._settings = settings
        self._chaos = chaos
        self._http = httpx.AsyncClient(timeout=30.0)

    async def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        if self._chaos is not None:
            await self._chaos.step("nvidia.rerank")
        body: dict[str, Any] = {
            "model": self._settings.rerank_model,
            "query": query,
            "passages": [{"text": doc} for doc in documents],
        }
        try:
            response = await self._http.post(
                self._settings.rerank_url,
                json=body,
                headers={"Authorization": f"Bearer {self._settings.api_key}"},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise InfraError(f"nvidia.rerank failed: {exc}") from exc
        payload: dict[str, Any] = response.json()
        scores = [0.0] * len(documents)
        for item in payload.get("results", []):
            scores[int(item["index"])] = float(item.get("relevance_score", 0.0))
        return scores
