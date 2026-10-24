"""Grounded answer composition with citations, PII/injection verification, reasoning trace."""

from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator
from typing import Any

import structlog

from atlas_core.config import AnswerSettings, SecuritySettings
from atlas_core.domain.models import Answer, Citation, Query, TraceStep
from atlas_core.domain.ports import ChatMessage, ChatPort, PiiDetectorPort
from atlas_core.security import HeuristicInjectionGuard, mask_text
from atlas_retrieval.retriever import RetrievalOutcome, Retriever

_log = structlog.get_logger(__name__)

_CITE_RE = re.compile(r"\[(\d+)\]")


class AnswerService:
    def __init__(
        self,
        settings: AnswerSettings,
        chat: ChatPort,
        retriever: Retriever,
        pii: PiiDetectorPort,
        guard: HeuristicInjectionGuard,
        security: SecuritySettings,
    ) -> None:
        self._settings = settings
        self._chat = chat
        self._retriever = retriever
        self._pii = pii
        self._guard = guard
        self._security = security

    async def answer(self, query: Query) -> Answer:
        outcome = await self._retriever.retrieve(query)
        context, citations = self._build_context(outcome)
        messages = [ChatMessage("system", self._settings.system_prompt + context), ChatMessage("user", query.text)]
        text = await self._chat.complete(messages)
        return self._finalize(
            text,
            citations,
            advantage_trace=outcome.trace,
            decompose_note=f"query decomposed, {len(outcome.contexts)} passages in context",
        )

    async def stream(self, query: Query) -> AsyncIterator[dict[str, Any]]:
        """SSE-friendly event stream: `delta` chunks, then one `final` event."""
        outcome = await self._retriever.retrieve(query)
        context, citations = self._build_context(outcome)
        messages = [ChatMessage("system", self._settings.system_prompt + context), ChatMessage("user", query.text)]
        parts: list[str] = []
        async for delta in self._chat.stream(messages):
            parts.append(delta)
            yield {"type": "delta", "text": delta}
        text = "".join(parts)
        answer = self._finalize(text, citations, outcome.trace, f"streamed {len(parts)} deltas")
        yield {"type": "final", "answer": answer.text, "citations": list(answer.citations)}  # type: ignore[list-item]

    def _build_context(self, outcome: RetrievalOutcome) -> tuple[str, list[Citation]]:
        blocks = [f"[{i + 1}] {ctx.text}\n(source={ctx.document_id})" for i, ctx in enumerate(outcome.contexts)]
        citations = [
            Citation(
                document_id=ctx.document_id,
                chunk_id=ctx.chunk_id,
                source_uri="",
                seq=i + 1,
                score=ctx.final_score,
            )
            for i, ctx in enumerate(outcome.contexts)
        ]
        return "\n\n".join(blocks), citations

    def _finalize(
        self,
        text: str,
        citations: list[Citation],
        advantage_trace: list[TraceStep],
        decompose_note: str,
    ) -> Answer:
        trace = [*advantage_trace, TraceStep("compose", decompose_note)]

        cited_numbers = {int(m) for m in _CITE_RE.findall(text)}
        kept = [c for c in citations if c.seq in cited_numbers] or citations

        if not self._security.disable_injection_guard:
            verdict = self._guard.scan(text)
            if verdict.risky:
                trace.append(TraceStep("verify", f"output flagged: {verdict.matches}"))
                _log.warning("answer.guard.flagged", matches=verdict.matches)

        findings = self._pii.detect(text) if self._security.pii_policy == "mask" else []
        if findings:
            text = mask_text(text, findings)
            trace.append(TraceStep("verify", f"masked {len(findings)} PII findings in output"))

        return Answer(text=text, citations=tuple(kept), trace=tuple(trace))


def new_query_id() -> uuid.UUID:
    return uuid.uuid4()
