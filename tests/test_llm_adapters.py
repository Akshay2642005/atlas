"""LangChain adapter wiring and behaviour (no network)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.documents import Document as LangChainDocument
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from atlas_core.config import AtlasSettings, ChaosSettings, NvidiaSettings
from atlas_core.domain.errors import InfraError
from atlas_core.domain.models import ChatMessage
from atlas_core.domain.ports import EmbedKind
from atlas_core.security.chaos import Chaos
from atlas_llm.anthropic_chat import AnthropicChat
from atlas_llm.groq_chat import GroqChat
from atlas_llm.langchain_chat import LangChainChat
from atlas_llm.nvidia_chat import NvidiaChat
from atlas_llm.nvidia_embed import NvidiaEmbedder
from atlas_llm.nvidia_rerank import NvidiaReranker
from atlas_llm.openai_chat import OpenAIChat
from atlas_llm.provider import build_chat

_PROVIDERS = {
    "groq": GroqChat,
    "openai": OpenAIChat,
    "anthropic": AnthropicChat,
    "nvidia": NvidiaChat,
}


@pytest.mark.parametrize(("provider", "expected"), _PROVIDERS.items())
def test_build_chat_selects_provider(provider: str, expected: type[LangChainChat]) -> None:
    settings = AtlasSettings(chat={"provider": provider})
    assert isinstance(build_chat(settings), expected)


def test_build_chat_uses_openai_compatible_base_url() -> None:
    settings = AtlasSettings(
        chat={"provider": "openai"},
        openai={"base_url": "http://localhost:11434/v1", "model": "qwen2.5"},
    )
    adapter = build_chat(settings)
    assert isinstance(adapter, OpenAIChat)
    assert adapter._model.openai_api_base == "http://localhost:11434/v1"  # pyright: ignore[reportPrivateUsage]


async def test_langchain_chat_complete_and_stream() -> None:
    model = FakeListChatModel(responses=["hello", "hello"])
    chat = LangChainChat(model, provider="test")

    assert await chat.complete([ChatMessage("user", "hi")]) == "hello"
    chunks = [chunk async for chunk in chat.stream([ChatMessage("user", "hi")])]
    assert "".join(chunks) == "hello"


async def test_langchain_chat_wraps_provider_errors() -> None:
    class _Broken(FakeListChatModel):
        async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("boom")

    chat = LangChainChat(_Broken(responses=[]), provider="broken")
    with pytest.raises(InfraError, match="broken.chat failed"):
        await chat.complete([ChatMessage("user", "hi")])


async def test_chaos_is_applied_before_chat_call() -> None:
    chat = LangChainChat(
        FakeListChatModel(responses=["unused"]),
        provider="test",
        chaos=Chaos(ChaosSettings(enabled=True, failure_rate=1.0)),
    )
    with pytest.raises(InfraError, match="chaos"):
        await chat.complete([ChatMessage("user", "hi")])


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Sequence[str]]] = []

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("passage", texts))
        return [[1.0, 0.0] for _ in texts]

    async def aembed_query(self, text: str) -> list[float]:
        self.calls.append(("query", [text]))
        return [0.0, 1.0]


async def test_embedder_routes_kind_to_langchain_methods() -> None:
    embedder = NvidiaEmbedder(NvidiaSettings())
    fake = _FakeEmbeddings()
    embedder._model = fake  # pyright: ignore[reportPrivateUsage]

    assert await embedder.embed([], EmbedKind.QUERY) == []
    assert await embedder.embed(["doc"], EmbedKind.PASSAGE) == [[1.0, 0.0]]
    assert await embedder.embed(["question"], EmbedKind.QUERY) == [[0.0, 1.0]]
    assert fake.calls == [("passage", ["doc"]), ("query", ["question"])]


class _FakeReranker:
    def model_copy(self, *, update: dict[str, Any]) -> _FakeReranker:
        self.top_n = update["top_n"]
        return self

    async def acompress_documents(
        self, documents: Sequence[LangChainDocument], query: str
    ) -> Sequence[LangChainDocument]:
        # Return in reverse order to prove scores are mapped by index, not position.
        return [
            document.model_copy(
                update={"metadata": {**document.metadata, "relevance_score": 1.0 / (rank + 1)}}
            )
            for rank, document in enumerate(reversed(documents))
        ]


async def test_reranker_maps_scores_back_to_input_order() -> None:
    reranker = NvidiaReranker(NvidiaSettings())
    reranker._model = _FakeReranker()  # pyright: ignore[reportPrivateUsage]

    scores = await reranker.rerank("query", ["a", "b", "c"])

    assert scores == pytest.approx([1 / 3, 1 / 2, 1.0])
    assert await reranker.rerank("query", []) == []


def test_chat_provider_rejects_unknown_value() -> None:
    with pytest.raises(ValueError):
        AtlasSettings(chat={"provider": "cohere"})
