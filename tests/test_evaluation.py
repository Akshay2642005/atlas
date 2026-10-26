"""Pure retrieval-evaluation maths."""

from __future__ import annotations

import math

import pytest

from atlas_retrieval.evaluation import (
    ArmSpec,
    EvalCase,
    RelevanceRef,
    bootstrap_delta,
    clean_question,
    normalize_question,
    parse_arm,
    parse_arms,
    percentile,
    resolve_case,
    score_ranking,
    summarize,
    truncate_embedding,
)


def test_recall_mrr_and_ndcg_for_a_hit() -> None:
    score = score_ranking(["x", "a", "b"], {"a"}, k=3)

    assert score.rank == 2
    assert score.recall == 1.0
    assert score.mrr == 0.5
    assert score.ndcg == pytest.approx(1 / math.log2(3))
    assert score.hit_at_1 == 0.0


def test_ndcg_rewards_higher_ranks() -> None:
    first = score_ranking(["a", "b", "c"], {"a"}, k=3)
    third = score_ranking(["b", "c", "a"], {"a"}, k=3)

    assert first.ndcg > third.ndcg
    assert first.ndcg == pytest.approx(1.0)


def test_multi_relevant_uses_ideal_dcg_for_the_relevant_count() -> None:
    score = score_ranking(["a", "x", "b"], {"a", "b"}, k=3)

    expected_dcg = 1.0 + 1 / math.log2(4)
    expected_ideal = 1.0 + 1 / math.log2(3)
    assert score.recall == 1.0
    assert score.ndcg == pytest.approx(expected_dcg / expected_ideal)


def test_recall_at_k_counts_only_the_cutoff() -> None:
    score = score_ranking(["x", "y", "a"], {"a"}, k=2)

    assert score.rank is None
    assert score.recall == 0.0
    assert score.ndcg == 0.0


def test_summarize_reads_latency_percentiles() -> None:
    scores = [score_ranking(["a"], {"a"}, k=1), score_ranking(["x"], {"a"}, k=1)]
    summary = summarize(scores, [10.0, 30.0])

    assert summary.count == 2
    assert summary.recall == 0.5
    assert summary.p50_ms == 20.0
    assert summary.p95_ms == pytest.approx(29.0)


def test_bootstrap_delta_is_zero_for_identical_arms() -> None:
    delta = bootstrap_delta([1.0, 0.5, 0.0], [1.0, 0.5, 0.0], resamples=200)

    assert (delta.mean, delta.low, delta.high) == (0.0, 0.0, 0.0)


def test_bootstrap_delta_reports_the_sign_of_a_consistent_gap() -> None:
    delta = bootstrap_delta([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], resamples=200)

    assert delta.mean == 1.0
    assert delta.low == 1.0
    assert delta.high == 1.0


def test_truncate_embedding_keeps_the_prefix_and_renormalizes() -> None:
    truncated = truncate_embedding([3.0, 4.0, 12.0], 2)

    assert truncated == pytest.approx([0.6, 0.8])
    assert math.sqrt(sum(value * value for value in truncated)) == pytest.approx(1.0)


def test_truncate_embedding_rejects_widening() -> None:
    with pytest.raises(ValueError, match="cannot truncate"):
        truncate_embedding([1.0, 0.0], 3)


def test_clean_question_prefers_a_question_line() -> None:
    raw = 'Here is a question:\n- "How does RRF fuse rankings?"\n'

    assert clean_question(raw) == "How does RRF fuse rankings?"
    assert normalize_question("How does RRF fuse rankings?") == "how does rrf fuse rankings"


def test_clean_question_falls_back_to_the_first_line() -> None:
    assert clean_question("Explain hybrid retrieval.") == "Explain hybrid retrieval."


def test_parse_arm_supports_presets_and_reembedding() -> None:
    assert parse_arm("full4096") == ArmSpec(name="full4096", kind="existing", dims=4096)
    assert parse_arm("trunc2048-hnsw") == ArmSpec(
        name="trunc2048-hnsw", kind="existing", dims=2048, ann=True
    )
    assert parse_arm("reembed:nvidia/nemotron-3.5-embed-1b@2048") == ArmSpec(
        name="reembed:nvidia/nemotron-3.5-embed-1b@2048",
        kind="reembed",
        dims=2048,
        model="nvidia/nemotron-3.5-embed-1b",
    )


def test_parse_arm_marks_pipeline_arms() -> None:
    hybrid = parse_arm("hybrid")
    reranked = parse_arm("hybrid-rerank")

    assert hybrid.kind == "hybrid"
    assert hybrid.is_pipeline
    assert reranked.kind == "hybrid-rerank"
    assert reranked.is_pipeline
    assert not parse_arm("full4096").is_pipeline


def test_parse_arm_reads_a_rerank_depth() -> None:
    assert parse_arm("hybrid-rerank@24").rerank_top_k == 24
    assert parse_arm("hybrid-rerank").rerank_top_k is None


def test_parse_arm_rejects_a_depth_on_plain_hybrid() -> None:
    with pytest.raises(ValueError, match="only hybrid-rerank"):
        parse_arm("hybrid@24")


def test_parse_arms_rejects_ann_above_the_cap() -> None:
    with pytest.raises(ValueError, match="4,000"):
        parse_arm("reembed:nvidia/nemotron-3.5-embed-8b@4096:hnsw")


def test_parse_arms_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        parse_arms("trunc2048,trunc2048")


def test_resolve_case_reports_missing_references() -> None:
    case = EvalCase(
        id="q1",
        query="How does RRF work?",
        relevant=(RelevanceRef("abc", 0), RelevanceRef("abc", 9)),
    )

    resolved, missing = resolve_case(case, {("abc", 0): "chunk-1"})

    assert resolved == ("chunk-1",)
    assert missing == 1


def test_percentile_interpolates() -> None:
    assert percentile([10.0, 20.0, 30.0], 50.0) == 20.0
    assert percentile([10.0, 20.0, 30.0], 95.0) == pytest.approx(29.0)
    assert percentile([], 95.0) == 0.0
