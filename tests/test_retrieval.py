"""Retrieval fusion + query decomposition."""

from __future__ import annotations

from atlas_retrieval.retriever import rrf_fusion, split_query


def test_rrf_prefers_documents_in_multiple_rankings() -> None:
    scores = rrf_fusion([["a", "b"], ["b", "c"]], k=60)
    assert scores["b"] > scores["a"] > scores["c"]
    assert scores["c"] > 0


def test_split_query_keeps_simple_queries_whole() -> None:
    assert split_query("What is our refund policy?") == ["What is our refund policy?"]


def test_split_query_decomposes_compound_questions() -> None:
    parts = split_query("What is the refund policy? How long does shipping take?")
    assert len(parts) == 2
