# Retrieval evaluation — Atlas corpus (hybrid)

Generated 2026-10-26 20:50:46 IST.

## Setup

- Tenant: `atlas-eval`
- Corpus: **112 chunks** from 20 documents, 4096 dims
- Queries: **40** scored, 0 skipped (ground truth no longer resolvable)
- Query set: `specs/eval/queries-atlas-repo.jsonl`
- Query embedder(s): `nvidia/nemotron-3.5-embed-8b`
- Top-10; Dense arms score cosine distance only; `hybrid` arms run production RRF
- Rerank coverage: `hybrid-rerank@50` 40/40, `hybrid-rerank@24` 40/40, `hybrid-rerank@12` 40/40 queries
- Server: `PostgreSQL 17.11 (Debian 17.11-1.pgdg12+2) on aarch64-unknown-linux-gnu, compiled by gcc (Debian 12.2.0-14+deb12u1) 12.2.0, 64-bit`; pgvector `0.8.7`

## Results

| arm | recall@10 | MRR | nDCG@10 | hit@1 | p50 | p95 | build |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full4096` | 92.5% | 0.642 | 71.2% | 45.0% | 2.2 ms | 2.4 ms | 0.0 s |
| `hybrid` | 97.5% | 0.569 | 66.7% | 35.0% | 48.0 ms | 52.0 ms | 0.0 s |
| `hybrid-rerank@50` | 100.0% | 0.808 | 85.7% | 67.5% | 1911.4 ms | 2015.8 ms | 0.0 s |
| `hybrid-rerank@24` | 100.0% | 0.808 | 85.7% | 67.5% | 1012.7 ms | 1058.9 ms | 0.0 s |
| `hybrid-rerank@12` | 97.5% | 0.802 | 84.6% | 67.5% | 638.1 ms | 791.6 ms | 0.0 s |

## Difference vs `full4096`

Paired bootstrap over queries (2,000 resamples, 95% interval). An interval
that includes zero is not distinguishable from query-sampling noise.

| arm | Δ recall | Δ nDCG |
| --- | ---: | ---: |
| `hybrid` recall | +5.0 pp | +0.0 … +12.5 pp |
| `hybrid` nDCG | -4.5 pp | -13.2 … +3.8 pp |
| `hybrid-rerank@50` recall | +7.5 pp | +0.0 … +15.0 pp |
| `hybrid-rerank@50` nDCG | +14.4 pp | +7.2 … +21.9 pp |
| `hybrid-rerank@24` recall | +7.5 pp | +0.0 … +15.0 pp |
| `hybrid-rerank@24` nDCG | +14.4 pp | +7.2 … +21.9 pp |
| `hybrid-rerank@12` recall | +5.0 pp | +0.0 … +12.5 pp |
| `hybrid-rerank@12` nDCG | +13.4 pp | +6.2 … +20.8 pp |

## Material regressions (missed, or dropped 3+ ranks)

- `hybrid`: Which container orchestration tool does Atlas use for its infrastructure? (rank 1 → 5)
- `hybrid`: What is the latency of the exact scan when processing 50,000 chunks? (rank 1 → 5)
- `hybrid`: What is the default model identifier specified in the NvidiaRerankSettings configuration? (rank 1 → 7)

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
  query embedding (cached here, so near zero) and, for `hybrid-rerank`, the
  hosted rerank round trip. That is the number users feel, but it is not a
  like-for-like comparison with dense SQL.

