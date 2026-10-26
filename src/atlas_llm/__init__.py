"""Atlas LLM adapters.

Adapter modules are re-exported lazily so importing the package (or starting the
API) does not load every provider SDK. `providers` and `build_chat` stay eager
because the composition root needs them.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from atlas_llm.provider import build_chat, providers

if TYPE_CHECKING:
    from atlas_llm.anthropic_chat import AnthropicChat
    from atlas_llm.groq_chat import GroqChat
    from atlas_llm.langchain_chat import LangChainChat
    from atlas_llm.nvidia_chat import NvidiaChat
    from atlas_llm.nvidia_embed import NvidiaEmbedder
    from atlas_llm.nvidia_rerank import NvidiaReranker
    from atlas_llm.openai_chat import OpenAIChat

__all__ = [
    "AnthropicChat",
    "GroqChat",
    "LangChainChat",
    "NvidiaChat",
    "NvidiaEmbedder",
    "NvidiaReranker",
    "OpenAIChat",
    "build_chat",
    "providers",
]

_LAZY: dict[str, tuple[str, str]] = {
    "AnthropicChat": ("atlas_llm.anthropic_chat", "AnthropicChat"),
    "GroqChat": ("atlas_llm.groq_chat", "GroqChat"),
    "LangChainChat": ("atlas_llm.langchain_chat", "LangChainChat"),
    "NvidiaChat": ("atlas_llm.nvidia_chat", "NvidiaChat"),
    "NvidiaEmbedder": ("atlas_llm.nvidia_embed", "NvidiaEmbedder"),
    "NvidiaReranker": ("atlas_llm.nvidia_rerank", "NvidiaReranker"),
    "OpenAIChat": ("atlas_llm.openai_chat", "OpenAIChat"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
