"""Transport-level checks: LangChain clients talk to the expected HTTP shapes.

A local fake provider serves OpenAI-compatible chat/embeddings, Anthropic
messages, and NVIDIA ranking responses. No external service or network is used.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from atlas_core.config import (
    AnthropicSettings,
    ChatSettings,
    GroqSettings,
    NvidiaSettings,
    OpenAISettings,
)
from atlas_core.domain.models import ChatMessage
from atlas_core.domain.ports import EmbedKind
from atlas_llm.anthropic_chat import AnthropicChat
from atlas_llm.groq_chat import GroqChat
from atlas_llm.nvidia_chat import NvidiaChat
from atlas_llm.nvidia_embed import NvidiaEmbedder
from atlas_llm.nvidia_rerank import NvidiaReranker
from atlas_llm.openai_chat import OpenAIChat

_MESSAGES = [ChatMessage("system", "be brief"), ChatMessage("user", "hi")]


def _chunk(text: str) -> bytes:
    payload = {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/messages"):
            return self._anthropic(body)
        if self.path.endswith("/embeddings"):
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            return self._json(
                {
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": i, "embedding": [float(i), 0.5]}
                        for i in range(len(inputs))
                    ],
                    "model": body.get("model", "m"),
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            )
        if self.path.endswith("/ranking"):
            return self._json(
                {
                    "rankings": [
                        {"index": i, "logit": 1.0 / (i + 1)} for i in range(len(body["passages"]))
                    ]
                }
            )
        if self.path.endswith("/chat/completions"):
            return self._openai_chat(body)
        self.send_error(404)

    def _openai_chat(self, body: dict[str, Any]) -> None:
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(_chunk("hel") + _chunk("lo") + b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        self._json(
            {
                "id": "c1",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    def _anthropic(self, body: dict[str, Any]) -> None:
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            events = [
                (
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_1",
                            "type": "message",
                            "role": "assistant",
                            "model": "m",
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 1, "output_tokens": 0},
                        },
                    },
                ),
                (
                    "content_block_start",
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                ),
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "hello"},
                    },
                ),
                ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                (
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": 1},
                    },
                ),
                ("message_stop", {"type": "message_stop"}),
            ]
            for name, event in events:
                self.wfile.write(f"event: {name}\ndata: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()
            return
        self._json(
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "m",
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )

    def _json(self, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@contextmanager
def _server() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


async def test_all_chat_providers_complete_and_stream() -> None:
    chat = ChatSettings(max_tokens=64)
    with _server() as root:
        openai_root = f"{root}/v1"
        adapters = [
            GroqChat(GroqSettings(api_key="k", base_url=openai_root, model="m"), chat),
            OpenAIChat(OpenAISettings(api_key="k", base_url=openai_root, model="m"), chat),
            NvidiaChat(NvidiaSettings(api_key="k", base_url=openai_root, chat_model="m"), chat),
            AnthropicChat(AnthropicSettings(api_key="k", base_url=root, model="m"), chat),
        ]
        for adapter in adapters:
            assert await adapter.complete(_MESSAGES) == "hello"
            chunks = [chunk async for chunk in adapter.stream(_MESSAGES)]
            assert "".join(chunks) == "hello"


async def test_embedder_and_reranker_hit_nvidia_routes() -> None:
    with _server() as root:
        base = f"{root}/v1"
        embedder = NvidiaEmbedder(
            NvidiaSettings(api_key="k", base_url=base, embedding_model="m")
        )
        assert await embedder.embed(["a", "b"], EmbedKind.PASSAGE) == [
            [0.0, 0.5],
            [1.0, 0.5],
        ]
        assert await embedder.embed(["q"], EmbedKind.QUERY) == [[0.0, 0.5]]

        reranker = NvidiaReranker(
            NvidiaSettings(api_key="k", rerank_base_url=base, rerank_model="m")
        )
        assert await reranker.rerank("q", ["a", "b", "c"]) == [1.0, 0.5, 1 / 3]
