# ADR 0004 — Retrieval fusion, reranking, and the lexical branch

Status: accepted · 2026-10-28

## Context

Production retrieval is dense + lexical, fused with reciprocal-rank fusion
(RRF, k=60), then the fused head is reranked by the hosted
`Nemotron-3.5-Rerank-8B` and truncated to `final_top_k`. ADR 0003 settled the
embedding width on a 112-chunk corpus and left two questions open, because that
corpus could not answer them:

1. Does the lexical branch earn its keep once reranking is in place? On 112
   chunks `dense-rerank@24` and `hybrid-rerank@24` ranked all 40 queries
   identically, but every question was written from its own answer chunk and
   none of them was an exact identifier.
2. Is a narrower embedding (truncated 2048, or the 1B model at 2048) actually
   worse?

To answer them the harness gained a second corpus: pinned public documentation
(FastAPI 0.162.0, Django 6.1.2, Pydantic v2.14.1, pgvector v0.8.7), 929
documents and **8,990 chunks**, in tenant `atlas-eval-docs`. Public docs were
chosen deliberately so no private text reaches the hosted embedder.

Two query sets are scored against it:

- **120 generated questions** (`specs/eval/queries-public-docs.jsonl`), one per
  sampled chunk, the same construction as the repository set.
- **60 exact identifiers** (`specs/eval/queries-public-docs-identifiers.jsonl`),
  produced by `scripts/gen_identifier_queries.py`: a symbol that appears in at
  most three chunks of the corpus (`MultipleObjectMixin`, `ALLOWED_HOSTS`,
  `InvalidCacheKey`) is used *alone* as the query, with its chunk as ground
  truth. This is the query shape the lexical branch exists for.

## Evidence

Reports: `specs/verifications/EVAL-retrieval-public-docs.md` and
`specs/verifications/EVAL-retrieval-public-docs-identifiers.md`. Intervals are
paired bootstraps over queries.

### Width (120 questions)

| arm | recall@10 | nDCG@10 | MRR | Δ nDCG vs `full4096` |
| --- | ---: | ---: | ---: | --- |
| `full4096` | 80.8% | 64.4% | 0.591 | — |
| `trunc2048` | 80.0% | 63.4% | 0.580 | −1.0 pp [−2.8, +0.6] |
| `reembed:…embed-1b@2048` | 80.8% | 60.9% | 0.545 | **−3.5 pp [−6.9, −0.2]** |

Truncating 4096 → 2048 costs nothing measurable. The 1B model is measurably
worse on nDCG and only ties on recall, which is the same ordering NVIDIA's own
leaderboard shows (60.65 vs 57.80).

### The lexical branch

| query set | arm | recall@10 | nDCG@10 | hit@1 |
| --- | --- | ---: | ---: | ---: |
| questions | `full4096` | 80.8% | 64.4% | 47.5% |
| questions | `hybrid` | 68.3% | 45.4% | 26.7% |
| questions | `dense-rerank@24` | 81.7% | 72.4% | 60.0% |
| questions | `hybrid-rerank@24` | 80.0% | 71.3% | 60.0% |
| identifiers | `full4096` | 65.0% | 44.8% | 26.7% |
| identifiers | `hybrid` | 65.0% | 46.2% | 30.0% |
| identifiers | `dense-rerank@24` | 68.3% | 55.0% | 38.3% |
| identifiers | `hybrid-rerank@24` | 73.3% | 58.1% | 38.3% |

The branch points in opposite directions depending on the query:

- **Questions, reranked:** `hybrid-rerank@24` − `dense-rerank@24` is
  −1.7 pp recall [−5.0, +1.7] and −1.1 pp nDCG [−3.6, +0.7]. Not significant,
  but the point estimate is negative and it costs 152 ms.
- **Identifiers, reranked:** the same comparison is **+5.0 pp recall**
  [−1.7, +11.7] and **+3.0 pp nDCG** [−0.6, +7.1]. Not significant at 60
  queries either, but positive, and it is the difference between finding an
  exact symbol and not.
- **Questions, unreranked:** fusion is a disaster — −12.5 pp recall
  [−18.4, −6.7] and −19.1 pp nDCG [−24.4, −13.8] against dense alone.

### Why fusion hurts (mechanism)

Diagnostics over the 120 question queries, using the production
`semantic_top_k = 50` / `lexical_top_k = 50` / `rrf_k = 60`:

- **28.8%** of the fused top-24 — the rerank window — is lexical-only chunks
  that dense never returned.
- **15 of 120** queries lose a relevant chunk that dense alone had in its
  top 10.
- 4 relevant chunks are pushed out of the rerank window entirely.

The cause is not the lexical list's weight but the RRF boost for co-occurrence.
A chunk at dense rank 40 that also appears at lexical rank 1 scores
`1/101 + 1/61 = 0.0263`, beating a chunk at dense rank 10 (`1/71 = 0.0141`).
Lexical co-occurrence therefore promotes mediocre dense candidates above good
ones, and RRF's flat k=60 cannot tell a precise list from a noisy one.

Shrinking the branch does not fix it either — recall falls monotonically with
lexical depth, and **zero** question queries were saved by a chunk only the
lexical branch found:

| `lexical_top_k` | 0 | 5 | 10 | 25 | 50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| recall@10 | 80.8% | 75.8% | 71.7% | 70.0% | 68.3% |
| fused top-24 slots that are lexical-only | 0% | 13.4% | 28.1% | 34.0% | 28.8% |

Down-weighting the lexical list was measured too, and does not rescue it:
at weight 0.25 the question set recovers to 76.7% recall / 54.6% nDCG (still
below dense-only's 80.8% / 64.4%), while identifiers fall to 61.7% / 44.4%
(below dense-only). Only the full weight recovers identifiers (65.0% / 46.2%).
The trade-off is real, not a tuning failure.

## Decision

1. **Keep 4096 dims and the 8B embedder.** Truncation to 2048 is a tie, not a
   win, so it does not justify a schema migration and a re-ingest; the 1B is
   measurably worse. ADR 0003 stands, now on 120 queries at 9k chunks rather
   than 40 at 112.
2. **Keep the lexical branch.** On question-shaped queries it is at worst
   neutral once reranking is in place (−1.1 pp nDCG, interval straddling zero),
   and on exact-identifier queries it is the only mechanism that recovers the
   target at all (+5 pp recall, +3 pp nDCG). Deleting it would trade an
   unproven regression for a proven loss on the query shape it exists for.
3. **Do not weight the branches.** Measured: every weight between 0.25 and 0.75
   is worse than both endpoints on both query sets.
4. **Reranking is load-bearing at this scale, not an optimisation.** It is the
   difference between 64.4% and 72.4% nDCG, and it is what contains the damage
   fusion does to the question set.

## Consequences

- The retriever's fail-open path is a liability. When the reranker is
  unreachable it serves the *fused* ordering for five minutes, which on this
  corpus is 19 pp nDCG worse than dense alone — worse than having no lexical
  branch at all. Serving the dense ordering during a rerank outage would be
  strictly better here, but it changes what `ENABLE_RERANK=false` means, so it
  is an open decision rather than part of this ADR.
- Fusion's co-occurrence boost is the thing to fix if the branch is to be
  improved: a cap on how far a co-occurrence can promote a candidate, or
  document-level deduplication before fusion, are the candidates. Neither is
  measured.
- Question generation still writes the question from the answer chunk, so the
  absolute numbers are optimistic. The identifier set is the honest one for the
  lexical question, and it is small (60 queries) — its intervals are wide by
  construction.
- `trunc2048` latency (51 ms vs 19 ms) is *not* evidence that 2048 is slower:
  that arm truncates and re-normalises per row at query time. Storage and I/O
  scaling for a real 2048 column is measured in `BENCH-indexing.md`, not here.
