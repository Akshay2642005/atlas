"""Register hosted NVIDIA models missing from the installed LangChain table.

`langchain-nvidia-ai-endpoints` ships a static model table and uses it both to
validate a model and to route hosted reranking to its retrieval endpoint. The
table in the pinned release predates Nemotron 3.5, so construction falls back to
a `/models` lookup and warns that the type is unknown. Registering the model
through the package's public `register_model` hook removes that lookup.
"""

from __future__ import annotations

import warnings

from langchain_nvidia_ai_endpoints import Model, register_model

HOSTED_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Hosted reranking is served from the retrieval API, not the
# OpenAI-compatible `/v1` root used for chat and embeddings.
HOSTED_RERANK_ENDPOINTS = {
    "nvidia/nemotron-3.5-rerank-8b": (
        "https://ai.api.nvidia.com/v1/retrieval/nvidia/nemotron-3.5-rerank-8b/reranking"
    ),
}

_registered: set[str] = set()


def _register_once(model: Model) -> None:
    if model.id in _registered:
        return
    with warnings.catch_warnings():
        # A future package release may ship this model already.
        warnings.filterwarnings("ignore", message=r"Model .* is already registered")
        register_model(model)
    _registered.add(model.id)


def is_hosted(base_url: str) -> bool:
    return base_url.rstrip("/") == HOSTED_BASE_URL


def register_hosted_embedding(model_id: str, base_url: str) -> None:
    """Register `model_id` as an embedding model on a hosted NIM root."""
    if not is_hosted(base_url):
        return
    _register_once(
        Model(
            id=model_id,
            model_type="embedding",
            client="NVIDIAEmbeddings",
            endpoint=f"{HOSTED_BASE_URL}/embeddings",
        )
    )


def register_hosted_rerank(model_id: str) -> str | None:
    """Register and return the hosted rerank endpoint for `model_id`."""
    endpoint = HOSTED_RERANK_ENDPOINTS.get(model_id)
    if endpoint is None:
        return None
    _register_once(
        Model(
            id=model_id,
            model_type="ranking",
            client="NVIDIARerank",
            endpoint=endpoint,
        )
    )
    return endpoint
