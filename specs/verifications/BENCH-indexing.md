# Indexing benchmark — vector(4096), halfvec(4096), halfvec(2048) + HNSW

Generated 2026-10-26 18:26:19 IST.

## Environment

- Server: `PostgreSQL 17.11 (Debian 17.11-1.pgdg12+2)`
- pgvector: `0.8.7`
- `shared_buffers`: `163848kB`
- `work_mem`: `4096kB`
- `maintenance_work_mem`: session `1GB` (server default `65536kB`)
- `max_parallel_workers_per_gather`: `2`
- `max_parallel_maintenance_workers`: session `0` (Docker `/dev/shm` is 64 MB; parallel builds fail with `No space left on device`)
- `effective_cache_size`: `5242888kB`
- HNSW parameters: `m = 16`, `ef_construction = 64`

Synthetic clustered vectors (64 clusters, noise 0.15), 100 queries, top-10. Latency is server-side `EXPLAIN ANALYZE` execution time.

| rows | vector(4096) exact p50 / p95 | halfvec(4096) exact p50 / p95 | halfvec(2048) exact p50 / p95 | HNSW ef=40 p50 / p95 | recall@10 | HNSW ef=100 p50 / p95 | recall@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 46.648 / 49.934 ms | 15.058 / 16.164 ms | 10.959 / 11.626 ms | 0.197 / 0.241 ms | 100.00% | 0.207 / 0.241 ms | 100.00% |
| 50,000 | 226.831 / 237.563 ms | 139.889 / 163.059 ms | 97.487 / 105.718 ms | 1.227 / 2.87 ms | 100.00% | 1.198 / 1.421 ms | 100.00% |
| 200,000 | 569.463 / 666.059 ms | 354.888 / 461.482 ms | 257.151 / 285.049 ms | 3.575 / 6.774 ms | 100.00% | 3.623 / 4.902 ms | 100.00% |

## Storage and build cost

| rows | vector(4096) | halfvec(4096) | halfvec(2048) | HNSW index | seed 4096 | seed 2048 | HNSW build |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 172.0 MB | 86.6 MB | 56.0 MB | 81.9 MB | 1.16 s | 0.31 s | 1.9 s |
| 50,000 | 859.4 MB | 432.4 MB | 279.6 MB | 409.6 MB | 5.34 s | 1.68 s | 17.02 s |
| 200,000 | 3437.4 MB | 1729.5 MB | 1118.2 MB | 1638.4 MB | 21.88 s | 7.75 s | 113.16 s |

## Plans

- 10,000: vector(4096) → Limit, Seq Scan, Sort; halfvec(4096) → Limit, Seq Scan, Sort; halfvec(2048) → Limit, Seq Scan, Sort; HNSW → Index Scan, Limit.
- 50,000: vector(4096) → Limit, Seq Scan, Sort; halfvec(4096) → Limit, Seq Scan, Sort; halfvec(2048) → Limit, Seq Scan, Sort; HNSW → Index Scan, Limit.
- 200,000: vector(4096) → Gather Merge, Limit, Seq Scan, Sort; halfvec(4096) → Gather Merge, Limit, Seq Scan, Sort; halfvec(2048) → Gather Merge, Limit, Seq Scan, Sort; HNSW → Index Scan, Limit.

## Caveat

Recall is measured against an exact `halfvec` scan over synthetic,
well-separated clusters. It is an upper bound: real embeddings overlap far
more, so production recall at the same `ef_search` will be lower. Timings
also reflect a warm cache and a single-tenant table; they do not include
filtering, joining, or network round-trip time.

`halfvec(4096)` is storable but cannot carry an HNSW index: pgvector caps
indexes at 4,000 dimensions for `halfvec`. It is included to separate the
cost of half precision from the cost of dropping dimensions.

