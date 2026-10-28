# Retrieval evaluation — Atlas corpus (public-docs-identifiers)

Generated 2026-10-28 07:27:09 IST.

## Setup

- Tenant: `atlas-eval-docs`
- Corpus: **8,990 chunks** from 929 documents, 4096 dims
- Queries: **60** scored, 0 skipped (ground truth no longer resolvable)
- Query set: `specs/eval/queries-public-docs-identifiers.jsonl`
- Query embedder(s): `nvidia/nemotron-3.5-embed-8b`
- Top-10; Dense arms score cosine distance only; `hybrid` arms run production RRF
- Rerank coverage: `hybrid-rerank@24` 60/60, `dense-rerank@24` 60/60 queries
- Server: `PostgreSQL 17.11 (Debian 17.11-1.pgdg12+2) on aarch64-unknown-linux-gnu, compiled by gcc (Debian 12.2.0-14+deb12u1) 12.2.0, 64-bit`; pgvector `0.8.7`

## Results

| arm | recall@10 | MRR | nDCG@10 | hit@1 | p50 | p95 | build |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full4096` | 65.0% | 0.385 | 44.8% | 26.7% | 19.8 ms | 21.5 ms | 0.0 s |
| `hybrid` | 65.0% | 0.404 | 46.2% | 30.0% | 80.9 ms | 102.4 ms | 0.0 s |
| `hybrid-rerank@24` | 73.3% | 0.529 | 58.1% | 38.3% | 1017.2 ms | 1171.9 ms | 0.0 s |
| `dense-rerank@24` | 68.3% | 0.506 | 55.0% | 38.3% | 982.1 ms | 1331.5 ms | 0.0 s |

## Difference vs `full4096`

Paired bootstrap over queries (2,000 resamples, 95% interval). An interval
that includes zero is not distinguishable from query-sampling noise.

| arm | Δ recall | Δ nDCG |
| --- | ---: | ---: |
| `hybrid` recall | +0.0 pp | -8.3 … +8.3 pp |
| `hybrid` nDCG | +1.3 pp | -5.5 … +8.1 pp |
| `hybrid-rerank@24` recall | +8.3 pp | +0.0 … +16.7 pp |
| `hybrid-rerank@24` nDCG | +13.3 pp | +6.8 … +19.9 pp |
| `dense-rerank@24` recall | +3.3 pp | +0.0 … +8.3 pp |
| `dense-rerank@24` nDCG | +10.2 pp | +4.5 … +16.4 pp |

## Material regressions (missed, or dropped 3+ ranks)

- `hybrid`: header_params (rank 7 → miss)
- `hybrid`: response_headers (rank 1 → 9)
- `hybrid`: references_model (rank 9 → miss)
- `hybrid`: timezone_name (rank 1 → 4)
- `hybrid`: file_object (rank 6 → miss)
- `hybrid`: script_code (rank 1 → miss)
- `hybrid`: year_in_school (rank 2 → 8)
- `hybrid-rerank@24`: header_params (rank 7 → miss)

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

