# Rerank latency vs candidate depth

Measured 2026-10-26 on the `atlas_eval` corpus (112 chunks, tenant `atlas-eval`)
against the hosted `nvidia/nemotron-3.5-rerank-8b` model.

## Why

`Retriever` fuses up to `semantic_top_k` (50) + `lexical_top_k` (50) candidates
and reranks them, then returns `final_top_k` (6). Rerank cost scales with the
number of documents sent to the model, so the tail of the fused list was being
paid for and then discarded. `RETRIEVAL__RERANK_TOP_K` now caps that list.

## Method

`scripts/bench_rerank_depth.py`:

1. take a judged query from `specs/eval/queries-atlas-repo.jsonl`;
2. build the fused candidate list with the production `rrf_fusion` (k=60) over
   `semantic_search` and `lexical_search` results;
3. time `NvidiaReranker.rerank` for several depths, 3 repetitions each.

Query vectors come from the local evaluation cache so the measurement does not
depend on the embedder endpoint.

## Results

| depth | median | min | max |
| ---: | ---: | ---: | ---: |
| 100 | 1.59 s | 1.59 s | 1.59 s |
| 50 | 1.62 s | 1.58 s | 1.66 s |
| 24 | 0.93 s | 0.91 s | 0.94 s |
| 12 | 0.53 s | 0.52 s | 0.54 s |
| 6 | 0.42 s | 0.42 s | 0.43 s |

Depth 100 and 50 are identical because this corpus produces only 50 unique
fused candidates.

## Reading

Cost is roughly linear in depth at ~30 ms per document plus ~0.4 s of fixed
round-trip overhead. Halving the candidate depth roughly halves the rerank
time:

- 50 → 24 candidates: **1.62 s → 0.93 s** (−43%).
- 50 → 12 candidates: **1.62 s → 0.53 s** (−67%).

For comparison, the dense exact scan over this corpus is ~2 ms and hybrid
retrieval without reranking is ~27 ms. Reranking is 20–60× the cost of the
retrieval it reorders, so the candidate count is the only lever that matters
for end-to-end latency.

## Limits

- One query, one 112-chunk corpus. The linear cost model should hold, but the
  per-document constant will vary with text length.
- This measures **latency only**.

## Outcome

The quality half ran later (`specs/verifications/EVAL-retrieval-hybrid.md`):
depth 24 ranked all 40 queries identically to depth 50, so
`RETRIEVAL__RERANK_TOP_K` defaults to **24** — half the p50 for the same
output. Depth 12 was rejected: it drops one query out of the top 10
(−2.5 pp recall, −1.1 pp nDCG), which is a real regression even though its
confidence interval touches zero.

That is why the benchmark depths are still 100/50/24/12/6: the curve is what
justifies the default, and re-running it on a different corpus is how the
default should be revisited.
