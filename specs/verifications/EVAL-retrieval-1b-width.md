# Retrieval evaluation — Atlas corpus (1b-width)

Generated 2026-10-26 19:34:26 IST.

## Setup

- Tenant: `atlas-eval`
- Corpus: **112 chunks** from 20 documents, 4096 dims
- Queries: **40** scored, 0 skipped (ground truth no longer resolvable)
- Query set: `specs/eval/queries-atlas-repo.jsonl`
- Query embedder(s): `nvidia/nemotron-3.5-embed-1b`, `nvidia/nemotron-3.5-embed-8b`
- Dense-only, top-10, cosine distance; no fusion or reranking
- Server: `PostgreSQL 17.11 (Debian 17.11-1.pgdg12+2) on aarch64-unknown-linux-gnu, compiled by gcc (Debian 12.2.0-14+deb12u1) 12.2.0, 64-bit`; pgvector `0.8.7`

## Results

| arm | recall@10 | MRR | nDCG@10 | hit@1 | p50 | p95 | build |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full4096` | 92.5% | 0.642 | 71.2% | 45.0% | 2.1 ms | 2.5 ms | 0.0 s |
| `trunc2048` | 92.5% | 0.630 | 70.2% | 45.0% | 2.0 ms | 2.6 ms | 0.0 s |
| `reembed:nvidia/nemotron-3.5-embed-1b@2048` | 92.5% | 0.648 | 71.7% | 47.5% | 0.9 ms | 1.0 ms | 0.1 s |
| `reembed:nvidia/nemotron-3.5-embed-1b@1024` | 95.0% | 0.632 | 71.0% | 45.0% | 0.6 ms | 0.7 ms | 0.0 s |

## Difference vs `full4096`

Paired bootstrap over queries (2,000 resamples, 95% interval). An interval
that includes zero is not distinguishable from query-sampling noise.

| arm | Δ recall | Δ nDCG |
| --- | ---: | ---: |
| `trunc2048` recall | +0.0 pp | +0.0 … +0.0 pp |
| `trunc2048` nDCG | -1.0 pp | -3.9 … +1.8 pp |
| `reembed:nvidia/nemotron-3.5-embed-1b@2048` recall | +0.0 pp | +0.0 … +0.0 pp |
| `reembed:nvidia/nemotron-3.5-embed-1b@2048` nDCG | +0.5 pp | -7.2 … +8.8 pp |
| `reembed:nvidia/nemotron-3.5-embed-1b@1024` recall | +2.5 pp | +0.0 … +7.5 pp |
| `reembed:nvidia/nemotron-3.5-embed-1b@1024` nDCG | -0.2 pp | -7.2 … +7.0 pp |

## Material regressions (missed, or dropped 3+ ranks)

- `trunc2048`: What vector dimension is assigned to each chunk when a document is ingested using the NVIDIA embedder configured with nemotron-3.5-embed-8b? (rank 4 → 7)
- `reembed:nvidia/nemotron-3.5-embed-1b@1024`: What is the default model specified for the GROQ provider? (rank 2 → 5)

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

