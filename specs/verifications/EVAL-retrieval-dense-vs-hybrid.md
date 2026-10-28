# Retrieval evaluation — Atlas corpus (dense-vs-hybrid)

Generated 2026-10-28 06:05:25 IST.

## Setup

- Tenant: `atlas-eval`
- Corpus: **112 chunks** from 20 documents, 4096 dims
- Queries: **40** scored, 0 skipped (ground truth no longer resolvable)
- Query set: `specs/eval/queries-atlas-repo.jsonl`
- Query embedder(s): `nvidia/nemotron-3.5-embed-8b`
- Top-10; Dense arms score cosine distance only; `hybrid` arms run production RRF
- Rerank coverage: `dense-rerank@24` 40/40, `hybrid-rerank@24` 40/40 queries
- Server: `PostgreSQL 17.11 (Debian 17.11-1.pgdg12+2) on aarch64-unknown-linux-gnu, compiled by gcc (Debian 12.2.0-14+deb12u1) 12.2.0, 64-bit`; pgvector `0.8.7`

## Results

| arm | recall@10 | MRR | nDCG@10 | hit@1 | p50 | p95 | build |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense-rerank@24` | 100.0% | 0.808 | 85.7% | 67.5% | 1039.4 ms | 1551.7 ms | 0.0 s |
| `hybrid-rerank@24` | 100.0% | 0.808 | 85.7% | 67.5% | 1056.2 ms | 1485.9 ms | 0.0 s |

## Difference vs `dense-rerank@24`

Paired bootstrap over queries (2,000 resamples, 95% interval). An interval
that includes zero is not distinguishable from query-sampling noise.

| arm | Δ recall | Δ nDCG |
| --- | ---: | ---: |
| `hybrid-rerank@24` recall | +0.0 pp | +0.0 … +0.0 pp |
| `hybrid-rerank@24` nDCG | +0.0 pp | +0.0 … +0.0 pp |

## Caveats

- Questions are generated from the source chunk, so they reuse its
  vocabulary. Real user questions drift further from the passage, which
  makes this an optimistic estimate of absolute quality.
- Ground truth is the single source chunk. Overlapping chunks or passages
  that also answer the question count as misses, which understates recall.
- Corpus size bounds how far these numbers extrapolate; recall is easier on
  small corpora. Use the deltas between arms, not the absolute values.
- ANN arms force `enable_seqscan = off` so the HNSW index is actually used.
  Their latency is not directly comparable to exact arms on tiny corpora.
- Dense-arm latency measures the SQL query only. `hybrid` arms also include
  query embedding (cached here, so near zero) and, for reranked arms, the
  hosted rerank round trip. That is the number users feel, but it is not a
  like-for-like comparison with dense SQL.

