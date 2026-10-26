# ADR 0002 — LangChain provider adapters

Status: accepted · 2026-10-26

## Context

The first implementation called Groq and NVIDIA directly through the OpenAI SDK and a
hand-rolled `httpx` rerank client. That duplicated request/response handling per
provider, made additional chat providers expensive to add, and drifted from the
LangChain packages the rest of the ecosystem uses for message conversion and
streaming. The product also needed OpenAI-compatible, Anthropic, and NVIDIA chat
options without changing domain code.

## Decisions

1. **LangChain packages per provider, one domain port.** `langchain-groq`,
   `langchain-openai`, `langchain-anthropic`, and `langchain-nvidia-ai-endpoints`
   each build a `BaseChatModel`; `LangChainChat` adapts it to `ChatPort`. Message
   mapping, streaming, chaos injection, and error normalisation live in that one
   adapter, so providers cannot drift in behaviour.
2. **Provider choice is configuration.** `CHAT__PROVIDER` selects `groq` (default),
   `openai`, `anthropic`, or `nvidia` via `build_chat()` in `atlas_llm/provider.py`.
   `ChatSettings` owns sampling/retry policy shared by every provider.
3. **OpenAI compatibility is a base URL.** `OPENAI__BASE_URL` accepts any
   OpenAI-compatible `/v1` root; no separate adapter is needed for vLLM, Ollama,
   LiteLLM, or similar gateways.
4. **Embeddings and reranking stay on NVIDIA NIM.** `NVIDIAEmbeddings` supplies
   `input_type=passage|query` (the package's `aembed_documents` / `aembed_query`) and
   `NVIDIARerank` returns `relevance_score` per document. The adapter maps scores
   back by an injected `atlas_index` metadata key, because the reranker returns
   documents in score order, not input order.
5. **Retries belong to the client.** Each LangChain model is constructed with
   `max_retries=CHAT__MAX_RETRIES`; the adapters do not layer tenacity on top.
   Provider exceptions are wrapped in `InfraError` at the port boundary.
6. **Privacy defaults.** NVIDIA telemetry is disabled on chat, embeddings, and
   reranking clients.

## Consequences

- Chat provider swaps are additive (new package + one branch in `build_chat`).
- Per-call token limits use the provider's native parameter name
  (`max_tokens` vs `max_completion_tokens`); `LangChainChat` carries that detail.
- `NVIDIA__BASE_URL` must be the NIM root or `/v1`; the LangChain client derives
  the ranking route and rejects deeper endpoint paths.
- Reranking still requires a live NIM endpoint; unit tests exercise score mapping
  with a fake compressor, and the integration path needs a deployment smoke test.
