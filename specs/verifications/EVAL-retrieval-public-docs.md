# Retrieval evaluation — Atlas corpus (public-docs)

Generated 2026-10-28 07:16:47 IST.

## Setup

- Tenant: `atlas-eval-docs`
- Corpus: **8,990 chunks** from 929 documents, 4096 dims
- Queries: **120** scored, 0 skipped (ground truth no longer resolvable)
- Query set: `specs/eval/queries-public-docs.jsonl`
- Query embedder(s): `nvidia/nemotron-3.5-embed-1b`, `nvidia/nemotron-3.5-embed-8b`
- Top-10; Dense arms score cosine distance only; `hybrid` arms run production RRF
- Rerank coverage: `hybrid-rerank@24` 120/120, `dense-rerank@24` 120/120 queries
- Server: `PostgreSQL 17.11 (Debian 17.11-1.pgdg12+2) on aarch64-unknown-linux-gnu, compiled by gcc (Debian 12.2.0-14+deb12u1) 12.2.0, 64-bit`; pgvector `0.8.7`

## Results

| arm | recall@10 | MRR | nDCG@10 | hit@1 | p50 | p95 | build |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full4096` | 80.8% | 0.591 | 64.4% | 47.5% | 19.2 ms | 20.1 ms | 0.0 s |
| `trunc2048` | 80.0% | 0.580 | 63.4% | 45.0% | 51.4 ms | 52.1 ms | 0.0 s |
| `reembed:nvidia/nemotron-3.5-embed-1b@2048` | 80.8% | 0.545 | 60.9% | 40.0% | 12.7 ms | 13.0 ms | 533.1 s |
| `hybrid` | 68.3% | 0.383 | 45.4% | 26.7% | 189.3 ms | 252.7 ms | 0.0 s |
| `hybrid-rerank@24` | 80.0% | 0.684 | 71.3% | 60.0% | 1164.2 ms | 1433.3 ms | 0.0 s |
| `dense-rerank@24` | 81.7% | 0.693 | 72.4% | 60.0% | 1012.4 ms | 1396.6 ms | 0.0 s |

## Difference vs `full4096`

Paired bootstrap over queries (2,000 resamples, 95% interval). An interval
that includes zero is not distinguishable from query-sampling noise.

| arm | Δ recall | Δ nDCG |
| --- | ---: | ---: |
| `trunc2048` recall | -0.8 pp | -2.5 … +0.0 pp |
| `trunc2048` nDCG | -1.0 pp | -2.8 … +0.6 pp |
| `reembed:nvidia/nemotron-3.5-embed-1b@2048` recall | +0.0 pp | -3.3 … +3.3 pp |
| `reembed:nvidia/nemotron-3.5-embed-1b@2048` nDCG | -3.5 pp | -6.9 … -0.2 pp |
| `hybrid` recall | -12.5 pp | -18.4 … -6.7 pp |
| `hybrid` nDCG | -19.1 pp | -24.4 … -13.8 pp |
| `hybrid-rerank@24` recall | -0.8 pp | -2.5 … +0.0 pp |
| `hybrid-rerank@24` nDCG | +6.9 pp | +3.8 … +10.3 pp |
| `dense-rerank@24` recall | +0.8 pp | -1.7 … +4.2 pp |
| `dense-rerank@24` nDCG | +8.0 pp | +4.4 … +12.0 pp |

## Material regressions (missed, or dropped 3+ ranks)

- `trunc2048`: Which attribute of Django’s generic UpdateView specifies the model fields to include in its form? (rank 10 → miss)
- `reembed:nvidia/nemotron-3.5-embed-1b@2048`: What type of server must be used to obtain performance benefits from Django's async views? (rank 1 → 4)
- `reembed:nvidia/nemotron-3.5-embed-1b@2048`: Which Django model field is listed as not represented in the form? (rank 5 → 9)
- `reembed:nvidia/nemotron-3.5-embed-1b@2048`: Which Django admin CSS classes were removed in favor of the “:nth-child(odd)” and “:nth-child(even)” pseudo‑classes? (rank 7 → miss)
- `reembed:nvidia/nemotron-3.5-embed-1b@2048`: What is the default alias used for the Avg aggregation function in Django’s ORM? (rank 1 → 5)
- `reembed:nvidia/nemotron-3.5-embed-1b@2048`: In Django's QuerySet API, what does the first() method return when the queryset matches no objects? (rank 1 → 8)
- `reembed:nvidia/nemotron-3.5-embed-1b@2048`: Which Python web framework, originally built on asyncio and designed to resemble Flask, uses uvloop instead of the default asyncio event loop? (rank 1 → miss)
- `reembed:nvidia/nemotron-3.5-embed-1b@2048`: In Django’s ORM, what values can the isnull field lookup take, and what SQL condition does each value generate? (rank 1 → 7)
- `hybrid`: Which pull request removed the slim package stub? (rank 2 → 7)
- `hybrid`: What is the default widget used for Django’s UUIDField? (rank 3 → 9)
- …and 37 more in the JSON report

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

