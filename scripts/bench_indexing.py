"""Benchmark exact vector search against half precision and an HNSW index.

The script seeds a throwaway database with synthetic clustered vectors, then
measures server-side query latency with EXPLAIN ANALYZE. It compares:

- vector(4096) exact scan        — today's configuration
- halfvec(4096) exact scan       — same vectors, half precision
- halfvec(2048) exact scan       — fewer dimensions, half precision
- halfvec(2048) HNSW             — approximate index

HNSW recall@10 is measured against an exact halfvec(2048) scan.

Run with the benchmark database already created:

    docker exec atlas-postgres-1 createdb -U atlas atlas_bench
    uv run --with "psycopg[binary]" --with numpy python scripts/bench_indexing.py

The database is disposable: the script drops and recreates its own tables and
refuses to run while another benchmark holds the advisory lock.
"""

from __future__ import annotations

import argparse
import json
import secrets
import statistics
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import psycopg

PGCOPY_HEADER = b"PGCOPY\n\xff\r\n\x00" + struct.pack(">II", 0, 0)
PGCOPY_TRAILER = struct.pack(">h", -1)

DEFAULT_DSN = "postgresql://atlas:atlas@127.0.0.1:5432/atlas_bench"
FULL_DIM = 4096
HALF_DIM = 2048
CLUSTERS = 64
NOISE = 0.15
QUERIES = 100
TOP_K = 10
EF_SEARCH = (40, 100)


def _copy_header() -> bytes:
    return PGCOPY_HEADER


def _vector_payload(vector: np.ndarray, half: bool) -> bytes:
    # numpy ships no type information here, so its `tobytes()` is `Any`.
    dtype = ">f2" if half else ">f4"
    body = bytes(vector.astype(dtype).tobytes())
    return struct.pack(">hh", int(vector.size), 0) + body


def _copy_chunk(
    cur: psycopg.Cursor[Any],
    table: str,
    start_id: int,
    vectors: np.ndarray,
    *,
    half: bool,
) -> None:
    parts: list[bytes] = [_copy_header()]
    for offset, vector in enumerate(vectors):
        payload = _vector_payload(vector, half)
        row_id = struct.pack(">q", start_id + offset)
        parts.append(struct.pack(">h", 2))
        parts.append(struct.pack(">i", len(row_id)))
        parts.append(row_id)
        parts.append(struct.pack(">i", len(payload)))
        parts.append(payload)
    parts.append(PGCOPY_TRAILER)
    with cur.copy(f"COPY {table} (id, embedding) FROM STDIN WITH (FORMAT binary)") as copy:
        copy.write(b"".join(parts))


def _create_table(cur: psycopg.Cursor[Any], table: str, dim: int, *, half: bool) -> None:
    column = f"halfvec({dim})" if half else f"vector({dim})"
    cur.execute(f"DROP TABLE IF EXISTS {table}")
    cur.execute(
        f"CREATE UNLOGGED TABLE {table} (id bigint PRIMARY KEY, embedding {column} NOT NULL)"
    )


def _claim_run(cur: psycopg.Cursor[Any]) -> str:
    """Take the benchmark lock and return a unique table suffix.

    Two concurrent runs would create and drop the same table names, so one run
    can delete another run's table mid-seed. The advisory lock serializes runs
    and the suffix keeps a stale table from an earlier crash from colliding.
    """
    cur.execute("SELECT pg_try_advisory_lock(hashtext('atlas-bench-indexing'))")
    row = cur.fetchone()
    if not row or not row[0]:
        raise SystemExit(
            "another indexing benchmark holds the advisory lock; wait for it to finish"
        )

    cur.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename LIKE 'bench\\_%'"
    )
    stale = [str(row[0]) for row in cur.fetchall()]
    for table in stale:
        cur.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')
    if stale:
        print(f"dropped {len(stale)} stale benchmark tables", flush=True)

    return secrets.token_hex(4)


def _seed(
    cur: psycopg.Cursor[Any],
    table: str,
    rows: int,
    dim: int,
    centers: np.ndarray,
    rng: np.random.Generator,
    *,
    half: bool,
    chunk_size: int = 5_000,
) -> None:
    for start in range(0, rows, chunk_size):
        end = min(start + chunk_size, rows)
        cluster_ids = rng.integers(0, len(centers), end - start)
        vectors = centers[cluster_ids] + rng.normal(scale=NOISE, size=(end - start, dim)).astype(
            np.float32
        )
        _copy_chunk(cur, table, start + 1, vectors, half=half)


def _literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{value:.6f}" for value in vector.tolist()) + "]"


def _plan_nodes(plan: dict[str, Any]) -> list[str]:
    nodes = [str(plan.get("Node Type"))]
    for child in plan.get("Plans", []):
        nodes.extend(_plan_nodes(child))
    return nodes


def _explain(
    cur: psycopg.Cursor[Any], table: str, query: str, cast: str, limit: int
) -> tuple[float, list[str]]:
    cur.execute(
        f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) "
        f"SELECT id FROM {table} ORDER BY embedding <=> %s::{cast} LIMIT {limit}",
        (query,),
    )
    payload = cur.fetchone()
    assert payload is not None
    result = payload[0][0]
    return float(result["Execution Time"]), _plan_nodes(result["Plan"])


def _top_ids(
    cur: psycopg.Cursor[Any],
    table: str,
    query: str,
    cast: str,
    *,
    ef_search: int | None = None,
    force_seq: bool = False,
    limit: int = TOP_K,
) -> list[int]:
    if force_seq:
        cur.execute("SET enable_indexscan = off; SET enable_bitmapscan = off")
    if ef_search is not None:
        cur.execute(f"SET hnsw.ef_search = {ef_search}")
    try:
        cur.execute(
            f"SELECT id FROM {table} ORDER BY embedding <=> %s::{cast} LIMIT {limit}",
            (query,),
        )
        return [row[0] for row in cur.fetchall()]
    finally:
        if force_seq:
            cur.execute("RESET enable_indexscan; RESET enable_bitmapscan")
        if ef_search is not None:
            cur.execute("RESET hnsw.ef_search")


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return ordered[index]


def _measure(
    cur: psycopg.Cursor[Any],
    table: str,
    queries: list[str],
    cast: str,
    *,
    ef_search: int | None = None,
    force_seq: bool = False,
) -> dict[str, Any]:
    if force_seq:
        cur.execute("SET enable_indexscan = off; SET enable_bitmapscan = off")
    if ef_search is not None:
        cur.execute(f"SET hnsw.ef_search = {ef_search}")

    for query in queries[:3]:
        _explain(cur, table, query, cast, TOP_K)

    times: list[float] = []
    nodes: set[str] = set()
    for query in queries:
        elapsed, plan_nodes = _explain(cur, table, query, cast, TOP_K)
        times.append(elapsed)
        nodes.update(plan_nodes)

    cur.execute("RESET enable_indexscan; RESET enable_bitmapscan")
    if ef_search is not None:
        cur.execute("RESET hnsw.ef_search")

    return {
        "p50_ms": round(statistics.median(times), 3),
        "p95_ms": round(_percentile(times, 0.95), 3),
        "mean_ms": round(statistics.fmean(times), 3),
        "plan_nodes": sorted(nodes),
    }


def _table_sizes(cur: psycopg.Cursor[Any], table: str, index: str | None = None) -> dict[str, int]:
    # pg_table_size excludes indexes, so a table's HNSW index is not counted
    # twice when it is reported separately.
    cur.execute("SELECT pg_table_size(%s::regclass)", (table,))
    row = cur.fetchone()
    assert row is not None
    sizes = {"table_bytes": int(row[0])}
    if index is not None:
        cur.execute("SELECT pg_relation_size(%s::regclass)", (index,))
        row = cur.fetchone()
        sizes["index_bytes"] = int(row[0]) if row else 0
    return sizes


def _recall(truth: list[list[int]], predicted: list[list[int]]) -> float:
    scores = [
        len(set(expected) & set(actual)) / len(expected)
        for expected, actual in zip(truth, predicted, strict=True)
    ]
    return round(statistics.fmean(scores), 4)


def _environment(cur: psycopg.Cursor[Any]) -> dict[str, Any]:
    """Capture the server configuration that shapes these timings."""
    cur.execute("SELECT version()")
    row = cur.fetchone()
    assert row is not None
    version = str(row[0])

    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    row = cur.fetchone()
    pgvector = str(row[0]) if row else None

    settings: dict[str, str] = {}
    for name in (
        "shared_buffers",
        "maintenance_work_mem",
        "max_parallel_workers_per_gather",
        "max_parallel_maintenance_workers",
        "work_mem",
        "effective_cache_size",
    ):
        cur.execute(
            "SELECT setting || coalesce(unit, '') FROM pg_settings WHERE name = %s",
            (name,),
        )
        row = cur.fetchone()
        settings[name] = str(row[0]) if row else "unknown"

    cur.execute(
        "SELECT current_setting('server_version_num')::int, "
        "coalesce((SELECT setting FROM pg_settings WHERE name = 'huge_pages'), 'unknown')"
    )
    row = cur.fetchone()
    assert row is not None

    return {
        "server_version": version.split(" on ")[0],
        "server_version_num": int(row[0]),
        "pgvector_version": pgvector,
        "huge_pages": str(row[1]),
        "settings": settings,
    }


def _persist(results: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bench-indexing.json").write_text(json.dumps(results, indent=2) + "\n")
    (out_dir / "BENCH-indexing.md").write_text(_markdown(results) + "\n")


def run(rows_list: list[int], dsn: str, out_dir: Path) -> dict[str, Any]:
    results: dict[str, Any] = {
        "config": {
            "rows": rows_list,
            "full_dim": FULL_DIM,
            "half_dim": HALF_DIM,
            "clusters": CLUSTERS,
            "noise": NOISE,
            "queries": QUERIES,
            "top_k": TOP_K,
            "ef_search": list(EF_SEARCH),
        },
        "sizes": [],
    }

    with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        results["environment"] = _environment(cur)
        run_id = _claim_run(cur)
        results["config"]["run_id"] = run_id
        cur.execute("SET maintenance_work_mem = '1GB'")
        # Docker defaults /dev/shm to 64 MB; parallel HNSW builds request a
        # shared segment as large as maintenance_work_mem and fail with
        # "No space left on device". Build single-process instead.
        cur.execute("SET max_parallel_maintenance_workers = 0")

        for rows in rows_list:
            size_result: dict[str, Any] = {"rows": rows}
            vec_rng = np.random.default_rng(1000 + rows)
            half_rng = np.random.default_rng(2000 + rows)
            query_rng = np.random.default_rng(3000 + rows)

            full_centers = vec_rng.normal(size=(CLUSTERS, FULL_DIM)).astype(np.float32)
            half_centers = half_rng.normal(size=(CLUSTERS, HALF_DIM)).astype(np.float32)

            full_queries = [
                _literal(
                    full_centers[i]
                    + query_rng.normal(scale=NOISE, size=FULL_DIM).astype(np.float32)
                )
                for i in query_rng.integers(0, CLUSTERS, QUERIES)
            ]
            half_queries = [
                _literal(
                    half_centers[i]
                    + query_rng.normal(scale=NOISE, size=HALF_DIM).astype(np.float32)
                )
                for i in query_rng.integers(0, CLUSTERS, QUERIES)
            ]

            full_table = f"bench_v4096_{rows}_{run_id}"
            h4096_table = f"bench_h4096_{rows}_{run_id}"
            half_table = f"bench_h2048_{rows}_{run_id}"
            half_index = f"{half_table}_hnsw"

            print(f"[{rows}] seeding vector({FULL_DIM}) ...", flush=True)
            started = time.perf_counter()
            _create_table(cur, full_table, FULL_DIM, half=False)
            _seed(cur, full_table, rows, FULL_DIM, full_centers, vec_rng, half=False)
            cur.execute(f"ANALYZE {full_table}")
            size_result["seed_v4096_seconds"] = round(time.perf_counter() - started, 2)
            full_sizes = _table_sizes(cur, full_table)
            size_result["v4096_table_bytes"] = full_sizes["table_bytes"]
            print(f"[{rows}] full seed: {size_result['seed_v4096_seconds']}s", flush=True)

            size_result["v4096_exact"] = _measure(cur, full_table, full_queries, "vector")

            # Same vectors, half precision. This separates the effect of storage
            # width from the effect of dropping dimensions.
            cur.execute(f"DROP TABLE {full_table}")
            print(f"[{rows}] seeding halfvec({FULL_DIM}) ...", flush=True)
            started = time.perf_counter()
            h4096_rng = np.random.default_rng(1000 + rows)
            h4096_centers = h4096_rng.normal(size=(CLUSTERS, FULL_DIM)).astype(np.float32)
            _create_table(cur, h4096_table, FULL_DIM, half=True)
            _seed(cur, h4096_table, rows, FULL_DIM, h4096_centers, h4096_rng, half=True)
            cur.execute(f"ANALYZE {h4096_table}")
            size_result["seed_h4096_seconds"] = round(time.perf_counter() - started, 2)
            h4096_sizes = _table_sizes(cur, h4096_table)
            size_result["h4096_table_bytes"] = h4096_sizes["table_bytes"]
            print(f"[{rows}] halfvec(4096) seed: {size_result['seed_h4096_seconds']}s", flush=True)

            size_result["h4096_exact"] = _measure(
                cur, h4096_table, full_queries, "halfvec", force_seq=True
            )
            cur.execute(f"DROP TABLE {h4096_table}")

            print(f"[{rows}] seeding halfvec({HALF_DIM}) ...", flush=True)
            started = time.perf_counter()
            _create_table(cur, half_table, HALF_DIM, half=True)
            _seed(cur, half_table, rows, HALF_DIM, half_centers, half_rng, half=True)
            cur.execute(f"ANALYZE {half_table}")
            size_result["seed_h2048_seconds"] = round(time.perf_counter() - started, 2)
            print(f"[{rows}] half seed: {size_result['seed_h2048_seconds']}s", flush=True)

            print(f"[{rows}] building HNSW ...", flush=True)
            started = time.perf_counter()
            cur.execute(
                f"CREATE INDEX {half_index} ON {half_table} "
                "USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 64)"
            )
            cur.execute(f"ANALYZE {half_table}")
            size_result["hnsw_build_seconds"] = round(time.perf_counter() - started, 2)
            half_sizes = _table_sizes(cur, half_table, half_index)
            size_result["h2048_table_bytes"] = half_sizes["table_bytes"]
            size_result["hnsw_index_bytes"] = half_sizes["index_bytes"]
            print(f"[{rows}] HNSW build: {size_result['hnsw_build_seconds']}s", flush=True)

            size_result["h2048_exact"] = _measure(
                cur, half_table, half_queries, "halfvec", force_seq=True
            )

            truth = [
                _top_ids(cur, half_table, query, "halfvec", force_seq=True)
                for query in half_queries
            ]

            size_result["h2048_hnsw"] = {}
            for ef in EF_SEARCH:
                measured = _measure(cur, half_table, half_queries, "halfvec", ef_search=ef)
                predicted = [
                    _top_ids(cur, half_table, query, "halfvec", ef_search=ef)
                    for query in half_queries
                ]
                measured["recall_at_10"] = _recall(truth, predicted)
                size_result["h2048_hnsw"][f"ef_search_{ef}"] = measured
                print(
                    f"[{rows}] HNSW ef={ef}: p50={measured['p50_ms']}ms "
                    f"p95={measured['p95_ms']}ms recall@10={measured['recall_at_10']}",
                    flush=True,
                )

            # Release the large tables before moving on: 200k x 4096 is 3.2 GB
            # and the next size should start with a clean disk budget.
            cur.execute(f"DROP TABLE IF EXISTS {full_table}")
            cur.execute(f"DROP TABLE IF EXISTS {h4096_table}")
            cur.execute(f"DROP TABLE IF EXISTS {half_table}")

            results["sizes"].append(size_result)
            _persist(results, out_dir)

    return results


def _markdown(results: dict[str, Any]) -> str:
    config = results["config"]
    env = results.get("environment", {})
    settings = env.get("settings", {})
    lines = [
        "# Indexing benchmark — vector(4096), halfvec(4096), halfvec(2048) + HNSW",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S %Z')}.",
        "",
        "## Environment",
        "",
        f"- Server: `{env.get('server_version', 'unknown')}`",
        f"- pgvector: `{env.get('pgvector_version', 'unknown')}`",
        f"- `shared_buffers`: `{settings.get('shared_buffers', 'unknown')}`",
        f"- `work_mem`: `{settings.get('work_mem', 'unknown')}`",
        f"- `maintenance_work_mem`: session `1GB` "
        f"(server default `{settings.get('maintenance_work_mem', 'unknown')}`)",
        f"- `max_parallel_workers_per_gather`: "
        f"`{settings.get('max_parallel_workers_per_gather', 'unknown')}`",
        "- `max_parallel_maintenance_workers`: session `0` "
        "(Docker `/dev/shm` is 64 MB; parallel builds fail with `No space left on device`)",
        f"- `effective_cache_size`: `{settings.get('effective_cache_size', 'unknown')}`",
        "- HNSW parameters: `m = 16`, `ef_construction = 64`",
        "",
        "Synthetic clustered vectors "
        f"({config['clusters']} clusters, noise {config['noise']}), "
        f"{config['queries']} queries, top-{config['top_k']}. "
        "Latency is server-side `EXPLAIN ANALYZE` execution time.",
        "",
        "| rows | vector(4096) exact p50 / p95 | halfvec(4096) exact p50 / p95 | "
        "halfvec(2048) exact p50 / p95 | HNSW ef=40 p50 / p95 | recall@10 | "
        "HNSW ef=100 p50 / p95 | recall@10 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for size in results["sizes"]:
        full = size["v4096_exact"]
        half_full = size["h4096_exact"]
        exact = size["h2048_exact"]
        ef40 = size["h2048_hnsw"]["ef_search_40"]
        ef100 = size["h2048_hnsw"]["ef_search_100"]
        lines.append(
            f"| {size['rows']:,} | {full['p50_ms']} / {full['p95_ms']} ms | "
            f"{half_full['p50_ms']} / {half_full['p95_ms']} ms | "
            f"{exact['p50_ms']} / {exact['p95_ms']} ms | "
            f"{ef40['p50_ms']} / {ef40['p95_ms']} ms | {ef40['recall_at_10']:.2%} | "
            f"{ef100['p50_ms']} / {ef100['p95_ms']} ms | {ef100['recall_at_10']:.2%} |"
        )

    lines += [
        "",
        "## Storage and build cost",
        "",
        "| rows | vector(4096) | halfvec(4096) | halfvec(2048) | HNSW index | "
        "seed 4096 | seed 2048 | HNSW build |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for size in results["sizes"]:
        lines.append(
            f"| {size['rows']:,} | {size['v4096_table_bytes'] / 1e6:.1f} MB | "
            f"{size['h4096_table_bytes'] / 1e6:.1f} MB | "
            f"{size['h2048_table_bytes'] / 1e6:.1f} MB | {size['hnsw_index_bytes'] / 1e6:.1f} MB | "
            f"{size['seed_v4096_seconds']} s | {size['seed_h2048_seconds']} s | "
            f"{size['hnsw_build_seconds']} s |"
        )

    lines += [
        "",
        "## Plans",
        "",
    ]
    for size in results["sizes"]:
        lines.append(
            f"- {size['rows']:,}: vector(4096) → "
            f"{', '.join(size['v4096_exact']['plan_nodes'])}; "
            f"halfvec(4096) → {', '.join(size['h4096_exact']['plan_nodes'])}; "
            f"halfvec(2048) → {', '.join(size['h2048_exact']['plan_nodes'])}; "
            f"HNSW → {', '.join(size['h2048_hnsw']['ef_search_40']['plan_nodes'])}."
        )
    lines += [
        "",
        "## Caveat",
        "",
        "Recall is measured against an exact `halfvec` scan over synthetic,",
        "well-separated clusters. It is an upper bound: real embeddings overlap far",
        "more, so production recall at the same `ef_search` will be lower. Timings",
        "also reflect a warm cache and a single-tenant table; they do not include",
        "filtering, joining, or network round-trip time.",
        "",
        "`halfvec(4096)` is storable but cannot carry an HNSW index: pgvector caps",
        "indexes at 4,000 dimensions for `halfvec`. It is included to separate the",
        "cost of half precision from the cost of dropping dimensions.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--rows", type=int, nargs="+", default=[10_000, 50_000, 200_000])
    parser.add_argument("--out-dir", type=Path, default=Path("specs/verifications"))
    args = parser.parse_args()

    results = run(args.rows, args.dsn, args.out_dir)
    _persist(results, args.out_dir)
    print("\n" + _markdown(results))


if __name__ == "__main__":
    main()
