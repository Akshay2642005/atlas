"""Dishka provider for retrieval orchestration."""

from __future__ import annotations

from dishka import Provider, Scope, provide

from atlas_core.config import AtlasSettings
from atlas_core.domain.ports import (
    ChatPort,
    EmbedderPort,
    PiiDetectorPort,
    VectorRepositoryPort,
)
from atlas_core.security import HeuristicInjectionGuard
from atlas_llm.nvidia_rerank import NvidiaReranker
from atlas_retrieval.answerer import AnswerService
from atlas_retrieval.retriever import Retriever


class RetrievalProvider(Provider):
    @provide(scope=Scope.APP)
    def retriever(
        self,
        settings: AtlasSettings,
        vectors: VectorRepositoryPort,
        embedder: EmbedderPort,
        reranker: NvidiaReranker,
    ) -> Retriever:
        return Retriever(
            settings=settings.retrieval,
            vectors=vectors,
            embedder=embedder,
            reranker=reranker if settings.retrieval.enable_rerank else None,
        )

    @provide(scope=Scope.APP)
    def answer_service(
        self,
        settings: AtlasSettings,
        chat: ChatPort,
        retriever: Retriever,
        pii: PiiDetectorPort,
        guard: HeuristicInjectionGuard,
    ) -> AnswerService:
        return AnswerService(
            settings=settings.answer,
            chat=chat,
            retriever=retriever,
            pii=pii,
            guard=guard,
            security=settings.security,
        )


def providers() -> list[Provider]:
    return [RetrievalProvider()]
