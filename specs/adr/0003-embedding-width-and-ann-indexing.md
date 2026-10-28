# ADR 0003 — Embedding width and ANN indexing

Status: accepted · 2026-10-26

## Context

Atlas stores chunk embeddings in pgVector. The configured NVIDIA embedder
(`nvidia/nemotron-3.5-embed-8b`) produces 4096-dimensional vectors, and the
vector column and the embedder's `dimensions` setting must agree or inserts fail
at runtime.

pgvector 0.8 caps ANN indexes by width:

| storage type | max dimensions for HNSW / IVFFlat |
| --- | --- |
| `vector` | 2,000 |
| `halfvec` | 4,000 |
| `bit` | 64,000 |

A 4096-dimensional column therefore cannot carry an ANN index, whichever
storage type is used. The alternatives are all lossy or narrower:

- fewer dimensions (a native ≤2,000-dim model, or a Matryoshka truncation);
- a `bit` expression index via `binary_quantize(embedding)`, used only for
  candidate generation and followed by an exact cosine re-rank.

Three options were on the table:

1. keep 4096 dims and `vector(4096)`, exact scan (status quo at the time);
2. keep 4096 dims but store `halfvec(4096)`, exact scan (no re-embedding);
3. drop to 2048 dims so HNSW becomes possible.

## Measurements

`scripts/bench_indexing.py` (`make bench`) seeds synthetic clustered vectors
into a throwaway database and times server-side `EXPLAIN ANALYZE` execution for
100 queries, top-10. Full numbers: `specs/verifications/BENCH-indexing.md`.

At 200,000 chunks:

| setup | p50 / p95 | disk | HNSW build |
| --- | ---: | ---: | ---: |
| `vector(4096)` exact | 569 / 666 ms | 3.44 GB | — |
| `halfvec(4096)` exact | **355 / 461 ms** | **1.73 GB** | — |
| `halfvec(2048)` exact | 257 / 285 ms | 1.12 GB | — |
| `halfvec(2048)` + HNSW | **3.6 / 6.8 ms** | 2.76 GB (1.12 table + 1.64 index) | 113 s |

At 50,000 chunks `halfvec(4096)` is 140/163 ms; at 10,000 it is 15/16 ms.
HNSW recall@10 was 100% on the synthetic clusters, which is an optimistic upper
bound — real embeddings overlap far more, and recall already dipped to 97.9%
at 50k in an earlier run of the same harness.

Quality data from NVIDIA's published ViDoRe V3 text leaderboard:

| model | native dims | NDCG@10 | warm latency | cold latency |
| --- | ---: | ---: | ---: | ---: |
| `nemotron-3.5-embed-8b` | 4096 (2048 truncatable) | 60.65 | 0.87 s | 77–120 s+ |
| `nemotron-3.5-embed-1b` | 2048 | 57.80 | 0.60 s | ~7 s |

Both models were probed live: the 1B returns 2048 dims, and the 8B honours
`dimensions: 2048`.

## Decision

Keep 4096 dimensions and the 8B embedder. Store the column as
`halfvec(4096)` and keep exact cosine search; do not add HNSW.

Migration `0003` alters `chunks.embedding` from `vector(4096)` to
`halfvec(4096)` with `USING embedding::halfvec(4096)`. It is an in-place cast:
same values, half the bytes, no re-ingestion. Migration `0002` (1024 → 4096)
did require re-ingestion because that change corrupted the representation
itself, not merely its storage.

Migration `0002` dropped the stale `chunks_hnsw` index, which pgvector could
not have built at 4096 dims. The GIN index on the generated `tsv` column is
unaffected, so hybrid retrieval keeps an indexed lexical side.

## Why not 2048 + HNSW

HNSW is dramatically faster at query time — three orders of magnitude on this
data — but it is not free:

- **Disk:** 2.76 GB vs 1.73 GB for `halfvec(4096)`, a 59% increase. Lower
  dimension does not pay for the index.
- **Writes:** every insert rewires the graph and is much more expensive than an
  append. Ingestion is the workload Atlas actually spends wall-clock on.
- **Operations:** `m`, `ef_construction`, `ef_search`, and recall monitoring all
  become things that can be misconfigured or regress silently.
- **Quality:** 2048 dims either means a weaker model (~3 NDCG points for the
  1B) or a truncation whose retrieval cost is unmeasured on Atlas's corpus.
- **The latency it buys is not the bottleneck.** The exact scan at 50k chunks
  is 140 ms; the LLM round trip and embedding call are seconds. Saving 140 ms
  to add a lossy index is a bad trade at this scale.

## Evaluation harness

The gate for a future width change is `scripts/eval_retrieval.py`, run through
`make eval-db`, `make eval-ingest`, `make eval-queries`, and `make eval`. It
compares arms on recall@k, MRR, nDCG@k, and latency, with a paired bootstrap
interval over queries. Ground truth is keyed by
`(document checksum, chunk seq)` so a query set survives re-ingestion.

Arms come in two families:

- `full4096`, `trunc2048`, `trunc2048-hnsw`, and
  `reembed:<model>@<dims>[:hnsw]` are dense-only SQL, so an embedding change is
  measured without fusion or reranking hiding it.
- `hybrid`, `hybrid-rerank[@<depth>]`, and `dense-rerank[@<depth>]` run the
  production `Retriever` (dense + lexical RRF, optionally reranked), so they
  measure the pipeline users actually hit, latency included. `dense-rerank`
  sets `lexical_top_k = 0`, which isolates what the lexical branch contributes
  once reranking is in place. The optional depth overrides
  `retrieval.rerank_top_k` for that arm. `make eval-pipeline` runs that
  comparison.

A first run over the repository corpus (112 chunks, 40 generated questions)
measured:

| arm | recall@10 | nDCG@10 | p50 |
| --- | ---: | ---: | ---: |
| `full4096` | 92.5% | 71.2% | 2.3 ms |
| `trunc2048` | 92.5% | 70.2% | 2.5 ms |
| `trunc2048-hnsw` | 92.5% | 70.2% | 0.9 ms |

The nDCG difference for truncation is **−1.0 pp (95% CI −3.9 … +1.8)**, which
is not distinguishable from query-sampling noise. The HNSW arm was verified to
use the index (`EXPLAIN` shows an index scan on the `halfvec_cosine_ops` index
under `enable_seqscan = off`).

The harness also validated a real model swap on the same query set
(`specs/verifications/EVAL-retrieval-1b-width.md`):

| arm | recall@10 | nDCG@10 |
| --- | ---: | ---: |
| `full4096` (8B) | 92.5% | 71.2% |
| 1B @ 2048 | 92.5% | 71.7% |
| 1B @ 1024 | 95.0% | 71.0% |

The 1B is not worse on this corpus within the confidence interval (+0.5 pp
nDCG, 95% CI −7.2 … +8.8). That does **not** overturn the decision: a 112-chunk
corpus has almost no headroom to show a difference, and retrieval is easy when
questions are generated from the target chunk. It does mean the 1B is not
obviously inferior here, so a tenant-scale run is the right way to settle it
rather than NVIDIA's leaderboard alone.

The harness is the gate, not this run. Reports:
`specs/verifications/EVAL-retrieval.md` and
`specs/verifications/EVAL-retrieval-1b-width.md`.

## Production pipeline: fusion, rerank, and the candidate cap

The dense arms answer "which embedding"; they cannot answer "is the pipeline
good", because production fuses and reranks. The `hybrid*` arms close that gap
by driving the real `Retriever`.

The first run of those arms produced a bug, not a result. `hybrid` scored
identically to `full4096` on every metric, which turned out to be
`lexical_search` using `plainto_tsquery` — an AND of every lexeme, so a
question-shaped query matched nothing and RRF fused a single list
(`specs/bugs/BUG-006.md`). With the lexical branch actually returning
candidates (median 50 per query), the same 40 queries measured:

| arm | recall@10 | nDCG@10 | MRR | hit@1 | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full4096` (dense only) | 92.5% | 71.2% | 0.642 | 45.0% | 2.2 ms | 2.4 ms |
| `hybrid` (RRF) | 97.5% | 66.7% | 0.569 | 35.0% | 48.0 ms | 52.0 ms |
| `hybrid-rerank@50` | 100.0% | 85.7% | 0.808 | 67.5% | 1911 ms | 2016 ms |
| `hybrid-rerank@24` | 100.0% | 85.7% | 0.808 | 67.5% | 1013 ms | 1059 ms |
| `hybrid-rerank@12` | 97.5% | 84.6% | 0.802 | 67.5% | 638 ms | 792 ms |
| `dense-rerank@24` | 100.0% | 85.7% | 0.808 | 67.5% | 1039 ms | 1552 ms |

Three findings, in order of how much they change the system:

1. **Reranking is the quality lever, not fusion.** Fusion alone *lowers*
   nDCG against dense (66.7% vs 71.2%, CI −13.2 … +3.8) while raising recall
   (97.5% vs 92.5%, CI 0 … +12.5): it pulls the right chunk into the candidate
   set but also pulls noise into the head. Rerank then fixes the order:
   +14.4 pp nDCG (CI +7.2 … +21.9), hit@1 45.0% → 67.5%, MRR 0.642 → 0.808.
2. **The reranker is fed far too much.** Capping candidates at 24 is
   *bit-identical* to 50 across all 40 queries (paired Δ = 0.00 on recall and
   nDCG) at 47% lower p50 (1013 ms vs 1911 ms). Depth 12 does cost quality:
   −2.5 pp recall and −1.1 pp nDCG, from exactly one query leaving the top 10.
3. **With reranking in place, the lexical branch adds nothing here.**
   `dense-rerank@24` and `hybrid-rerank@24` rank all 40 queries identically
   (paired Δ = 0.00 on both metrics) for 17 ms less per query. On a corpus of
   generated questions that reuse their source chunk's vocabulary, dense
   retrieval already surfaces the right candidates.

Decision: keep the pipeline as built, with `retrieval.rerank_top_k = 24`.

- Depth 24 is the smallest measured depth that loses nothing, and it is 4× the
  production `final_top_k` of 6, so there is headroom for the answer stage.
- The lexical branch is **retained**. It is the only thing raising recall in
  the unreranked path, and exact-identifier queries (error codes, SKUs, config
  keys) are exactly where dense retrieval fails and the corpus here has none.
  Its reranked contribution is unproven rather than disproven: this is a
  112-chunk corpus with questions written from the answer.
- Both findings are corpus-bounded. The depth cap is safe to adopt because it
  is a monotone reduction of work with an identical measured output; dropping
  the lexical branch would be a behaviour change justified by the same weak
  corpus, so it waits for the tenant-scale run.

Reports: `specs/verifications/EVAL-retrieval-hybrid.md`,
`specs/verifications/EVAL-retrieval-dense-vs-hybrid.md`, and
`specs/verifications/BENCH-rerank-depth.md`.

## Consequences

- Recall is exact; there is no approximate-index tuning to get wrong.
- Dense latency still grows linearly with a tenant's chunk count. At 10k/50k
  chunks it is comfortably below interactive budgets; at 200k it is ~355 ms.
- If scale demands ANN, revisit in this order:
  1. HNSW over 8B truncated to 2048 dims (already supported by the model);
  2. `binary_quantize` candidate generation with exact re-rank, which keeps
     full precision for scoring;
  3. only then a smaller embedder.

  Trigger conditions: a tenant above ~500k–1M chunks, or retrieval p95 above
  ~1 s, or a UI that needs sub-10 ms retrieval.
- Any of those changes must be gated on an evaluation harness over Atlas's own
  corpus. The quality figures above are NVIDIA's benchmarks, not ours, and the
  recall figures come from synthetic clusters.
- The embedding width is configuration, not schema magic. Changing
  `NVIDIA__EMBEDDING_DIMENSIONS` requires a matching migration.
- `nvidia/nv-embedqa-e5-v5` is EOL (410 Gone since 2026-08-25); the default is
  the 4096-native `nvidia/nemotron-3.5-embed-8b`.
- The 8B's cold start is operational, not architectural: warm it is 0.87 s.
  Keep it warm (scheduled ping) or self-host the NIM rather than trading
  retrieval quality for a cold-start symptom. The adapter widens the client
  read timeout to 180s and retries transient failures once.
- Reranking uses the hosted `nvidia/nemotron-3.5-rerank-8b` model. The hosted
  rerank API is served from `/v1/retrieval/<model>/reranking`, not the
  OpenAI-compatible `/v1` root, so the adapter registers that endpoint with
  `langchain-nvidia-ai-endpoints` instead of reusing `NVIDIA__BASE_URL`.
  Self-hosted NIM/vLLM deployments set `NVIDIA__RERANK_BASE_URL` and use the
  client's derived `/ranking` route.
- The retriever still treats rerank as optional: an unreachable endpoint is
  paused for five minutes instead of failing the request or retrying on every
  query.
- `retrieval.rerank_top_k` caps the fused list at **24** before reranking
  (default `final_top_k` is 6). This is the measured no-loss depth, not a
  guess: 50 candidates produced an identical ranking at nearly double the p50.
- Reranking dominates retrieval latency: ~1.0 s at depth 24 against ~26–48 ms
  for the hybrid SQL. The next latency lever is the reranker itself (a
  self-hosted NIM, batching, or a smaller rerank model), not the query path.
  Any candidate-depth reduction below 24 buys ~375 ms and costs a query.
- Reranking, not fusion, is what moved quality on the evaluation corpus. The
  lexical branch stays because it is the recall source without a reranker and
  the only defence against exact-identifier queries; that claim needs a
  tenant-scale corpus with drifted questions to be settled either way.
