"""Retrieval evaluation harness for Atlas's own corpus.

Subcommands:

    ingest    ingest a directory into an evaluation tenant
    generate  sample chunks and ask the chat model for one question per chunk
    run       compare retrieval arms on a query set and write a report

The harness is deliberately separate from the production path: it reads the
same `chunks` table, but materialises temporary tables in the `atlas_eval_eval`
schema and drops them afterwards unless `--keep-tables` is set.

Typical session:

    POSTGRES__URL=postgresql+asyncpg://atlas:atlas@localhost:5432/atlas_eval \\
      uv run python scripts/eval_retrieval.py ingest --path docs --path README.md
    POSTGRES__URL=... uv run python scripts/eval_retrieval.py generate --sample 40
    POSTGRES__URL=... uv run python scripts/eval_retrieval.py run \\
      --queries specs/eval/queries-atlas-repo.jsonl --arms full4096,trunc2048

Arms are dense-only on purpose: the question is whether the embedding width
still retrieves the same chunks, not whether fusion or reranking can hide a
weaker embedding.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
import re
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from atlas_core.config import AtlasSettings
from atlas_core.domain.errors import InfraError
from atlas_core.domain.models import ChatMessage, Query
from atlas_core.domain.ports import (
    ChatPort,
    ChunkerPort,
    EmbedderPort,
    EmbedKind,
    RerankerPort,
    VectorRepositoryPort,
)
from atlas_core.security.injection import HeuristicInjectionGuard
from atlas_core.security.pii import RegexPiiDetector
from atlas_core.security.sanitize import Sanitizer
from atlas_ingest.chunkers import RecursiveChunker, SemanticChunker
from atlas_ingest.loaders import (
    MIME_HTML,
    MIME_MARKDOWN,
    MIME_PDF,
    MIME_PLAIN,
    checksum,
)
from atlas_ingest.pipeline import IngestPipeline, IngestSpec
from atlas_llm.nvidia_embed import NvidiaEmbedder
from atlas_llm.nvidia_rerank import NvidiaReranker
from atlas_llm.provider import build_chat
from atlas_retrieval.evaluation import (
    ArmSpec,
    EvalCase,
    QueryScore,
    RelevanceRef,
    ScoreSummary,
    bootstrap_delta,
    clean_question,
    normalize_question,
    parse_arms,
    resolve_case,
    score_ranking,
    summarize,
    truncate_embedding,
)
from atlas_retrieval.retriever import Retriever, split_query
from atlas_vector.engine import create_engine_and_session
from atlas_vector.repositories import PgDocumentRepository, PgVectorRepository

DEFAULT_TENANT = "atlas-eval"
DEFAULT_ARMS = "full4096,trunc2048"
DEFAULT_EF_SEARCH = 40
EVAL_SCHEMA = "atlas_eval_eval"
DEFAULT_CACHE = Path(".eval-cache/embeddings.jsonl")
_MIME_BY_SUFFIX: dict[str, str] = {
    ".md": MIME_MARKDOWN,
    ".markdown": MIME_MARKDOWN,
    ".txt": MIME_PLAIN,
    ".html": MIME_HTML,
    ".htm": MIME_HTML,
    ".pdf": MIME_PDF,
}

_SYSTEM_PROMPT = (
    "You write retrieval-evaluation questions. Given one passage, write one "
    "specific, standalone question that the passage answers. The question must "
    "not mention 'the passage', 'the text', or 'above', and must be answerable "
    "without seeing the passage. Return only the question, one line."
)


# --------------------------------------------------------------------------- #
# Settings and database helpers
# --------------------------------------------------------------------------- #


def _settings(url: str | None) -> AtlasSettings:
    settings = AtlasSettings()
    if url is None:
        return settings
    postgres = settings.postgres.model_copy(update={"url": url})
    return settings.model_copy(update={"postgres": postgres})


async def _rows(
    conn: AsyncConnection, sql: str, params: Mapping[str, Any] | None = None
) -> list[Mapping[str, Any]]:
    result = await conn.execute(text(sql), dict(params or {}))
    return [dict(row) for row in result.mappings().all()]


async def _one(
    conn: AsyncConnection, sql: str, params: Mapping[str, Any] | None = None
) -> Mapping[str, Any]:
    rows = await _rows(conn, sql, params)
    if not rows:
        raise RuntimeError(f"query returned no rows: {sql.strip()[:80]}")
    return rows[0]


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{value:.7g}" for value in values) + "]"


def _log(message: str) -> None:
    print(message, flush=True)


class _EmbeddingCache:
    """Append-only cache of embedding vectors, keyed by model/width/kind/text.

    The hosted 8B embedder can 504 for minutes at a time. A cached query vector
    makes an evaluation reproducible and lets a rerun skip the flaky call
    entirely. The cache is local state and never committed.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._vectors: dict[str, list[float]] = {}
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                self._vectors[str(item["key"])] = [float(value) for value in item["vector"]]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

    def get(self, key: str) -> list[float] | None:
        return self._vectors.get(key)

    def put_many(self, items: Iterable[tuple[str, Sequence[float]]]) -> None:
        pending: list[tuple[str, list[float]]] = []
        for key, vector in items:
            values = [round(float(value), 7) for value in vector]
            self._vectors[key] = values
            pending.append((key, values))
        if not pending:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            for key, values in pending:
                handle.write(json.dumps({"key": key, "vector": values}) + "\n")


def _embed_key(model: str, dims: int, kind: EmbedKind, text: str) -> str:
    payload = f"{model}|{dims}|{kind.value}|{text}".encode()
    return hashlib.sha256(payload).hexdigest()


def _is_cold_start(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in ("504", "503", "timed out", "timeout", "temporarily unavailable")
    )


async def _embed_with_retry(
    embedder: EmbedderPort,
    texts: Sequence[str],
    kind: EmbedKind,
    label: str,
    *,
    attempts: int = 3,
    wait_seconds: float = 30.0,
) -> list[list[float]]:
    """Retry a cold NVIDIA NIM embedder.

    The hosted 8B model can spend longer than the server's 120 s ceiling on a
    cold start, so the adapter's immediate retry can also time out. Back off
    between attempts instead of failing a whole evaluation run.
    """
    for attempt in range(1, attempts + 1):
        try:
            return await embedder.embed(texts, kind=kind)
        except InfraError as exc:
            if attempt == attempts or not _is_cold_start(exc):
                raise SystemExit(f"{label} failed after {attempt} attempt(s): {exc}") from exc
            wait = wait_seconds * attempt
            _log(f"      {label}: cold start (attempt {attempt}); waiting {wait:.0f}s")
            await asyncio.sleep(wait)
    raise AssertionError("unreachable")


async def _embed_cached(
    embedder: EmbedderPort,
    texts: Sequence[str],
    kind: EmbedKind,
    label: str,
    *,
    model: str,
    dims: int,
    cache: _EmbeddingCache,
) -> list[list[float]]:
    keys = [_embed_key(model, dims, kind, text) for text in texts]
    missing = [
        (position, text)
        for position, (key, text) in enumerate(zip(keys, texts, strict=True))
        if cache.get(key) is None
    ]
    if missing:
        vectors = await _embed_with_retry(embedder, [text for _, text in missing], kind, label)
        cache.put_many(
            (keys[position], vector) for (position, _), vector in zip(missing, vectors, strict=True)
        )
    resolved: list[list[float]] = []
    for key in keys:
        vector = cache.get(key)
        if vector is None:
            raise RuntimeError(f"embedding cache miss for {key}")
        resolved.append(vector)
    return resolved


class _CachedEmbedder:
    """`EmbedderPort` that serves query vectors from the local cache.

    The production `Retriever` embeds each sub-query itself. Wrapping it keeps
    pipeline arms on the same cached vectors as the dense arms, so a flaky NIM
    endpoint cannot make the two incomparable or abort the run.
    """

    def __init__(
        self,
        embedder: EmbedderPort,
        *,
        model: str,
        dims: int,
        cache: _EmbeddingCache,
    ) -> None:
        self._embedder = embedder
        self._model = model
        self._dims = dims
        self._cache = cache

    async def embed(self, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]:
        return await _embed_cached(
            self._embedder,
            texts,
            kind,
            "retriever embedding",
            model=self._model,
            dims=self._dims,
            cache=self._cache,
        )


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #


def _display_path(path: Path) -> str:
    """Prefer a repository-relative path so committed query sets stay portable."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _collect_files(paths: Sequence[str], extensions: set[str]) -> list[Path]:
    found: dict[Path, None] = {}
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            if path.suffix.lower() not in extensions:
                raise SystemExit(f"unsupported file type: {path}")
            found[path.resolve()] = None
            continue
        if not path.is_dir():
            raise SystemExit(f"path not found: {path}")
        for candidate in sorted(path.rglob("*")):
            if not candidate.is_file() or candidate.suffix.lower() not in extensions:
                continue
            if any(part.startswith(".") for part in candidate.relative_to(path).parts):
                continue
            found[candidate.resolve()] = None
    return sorted(found)


async def _existing_checksums(engine: AsyncEngine, tenant: str) -> set[str]:
    async with engine.connect() as conn:
        rows = await _rows(
            conn, "SELECT checksum FROM documents WHERE tenant_id = :tenant", {"tenant": tenant}
        )
    return {str(row["checksum"]) for row in rows}


async def cmd_ingest(args: argparse.Namespace) -> int:
    settings = _settings(args.url)
    extensions = {suffix.strip().lower() for suffix in args.extensions.split(",") if suffix.strip()}
    for suffix in extensions:
        if suffix not in _MIME_BY_SUFFIX:
            raise SystemExit(
                f"unsupported extension {suffix!r}; supported: {sorted(_MIME_BY_SUFFIX)}"
            )
    files = _collect_files(args.path, extensions)
    if not files:
        raise SystemExit("no matching files found")

    engine, session_factory = create_engine_and_session(settings.postgres)
    embedder = NvidiaEmbedder(settings.nvidia)
    chunker: ChunkerPort
    if settings.chunking.enable_semantic_chunks:
        chunker = SemanticChunker(embedder, settings.chunking)
    else:
        chunker = RecursiveChunker(settings.chunking)
    pipeline = IngestPipeline(
        settings=settings,
        documents=PgDocumentRepository(session_factory),
        vectors=PgVectorRepository(session_factory),
        embedder=embedder,
        chunker=chunker,
        sanitizer=Sanitizer(),
        pii=RegexPiiDetector(),
        guard=HeuristicInjectionGuard(),
    )

    seen = await _existing_checksums(engine, args.tenant)
    ingested = skipped = failed = chunks = 0
    try:
        for path in files:
            content = path.read_bytes()
            digest = checksum(content)
            if digest in seen:
                skipped += 1
                _log(f"skip  {path} (already ingested)")
                continue
            mime = _MIME_BY_SUFFIX[path.suffix.lower()]
            try:
                result = await pipeline.run(
                    IngestSpec(
                        tenant_id=args.tenant,
                        content=content,
                        mime=mime,
                        source_uri=_display_path(path),
                    )
                )
            except Exception as exc:  # noqa: BLE001 — one bad file must not hide the rest
                failed += 1
                _log(f"fail  {path}: {exc}")
                continue
            seen.add(digest)
            ingested += 1
            chunks += result.chunks
            _log(f"ok    {path} ({result.chunks} chunks)")
    finally:
        await engine.dispose()

    _log(f"\ningested {ingested} document(s), {chunks} chunks; skipped {skipped}; failed {failed}")
    return 0 if ingested or skipped else 1


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #


async def _sample_chunks(
    engine: AsyncEngine, tenant: str, sample: int, min_chars: int, max_chars: int, seed: int
) -> list[Mapping[str, Any]]:
    async with engine.connect() as conn:
        return await _rows(
            conn,
            """
            SELECT c.id::text AS chunk_id,
                   c.seq,
                   c.text,
                   d.checksum,
                   d.source_uri
              FROM chunks c
              JOIN documents d ON d.id = c.document_id
             WHERE c.tenant_id = :tenant
               AND length(c.text) BETWEEN :min_chars AND :max_chars
             ORDER BY md5(c.id::text || :seed)
             LIMIT :sample
            """,
            {
                "tenant": tenant,
                "min_chars": min_chars,
                "max_chars": max_chars,
                "seed": str(seed),
                "sample": sample,
            },
        )


async def _ask_question(chat: ChatPort, passage: str) -> str:
    # Reasoning models (e.g. gpt-oss) spend part of the completion budget on
    # hidden reasoning. Too small a limit yields an empty `content` string even
    # though the call succeeds, so leave generous headroom for one question.
    response = await chat.complete(
        [
            ChatMessage(role="system", content=_SYSTEM_PROMPT),
            ChatMessage(role="user", content=f"Passage:\n\n{passage}\n\nQuestion:"),
        ],
        max_tokens=1024,
    )
    return clean_question(str(response))


def _generator_label(settings: AtlasSettings) -> str:
    provider = settings.chat.provider
    if provider == "groq":
        return f"groq:{settings.groq.model}"
    if provider == "openai":
        return f"openai:{settings.openai.model}"
    if provider == "anthropic":
        return f"anthropic:{settings.anthropic.model}"
    return f"nvidia:{settings.nvidia.chat_model}"


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)


async def cmd_generate(args: argparse.Namespace) -> int:
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"{args.out} exists; pass --overwrite to replace it")
    settings = _settings(args.url)
    engine, _ = create_engine_and_session(settings.postgres)
    try:
        rows = await _sample_chunks(
            engine, args.tenant, args.sample, args.min_chars, args.max_chars, args.seed
        )
    finally:
        await engine.dispose()
    if not rows:
        raise SystemExit(
            f"no chunks for tenant {args.tenant!r}; run `ingest` first or widen --min-chars"
        )

    chat = build_chat(settings)
    records: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    generated_at = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    for index, row in enumerate(rows, start=1):
        try:
            question = await _ask_question(chat, str(row["text"]))
        except Exception as exc:  # noqa: BLE001 — skip a bad generation, keep the set
            _log(f"skip  chunk {row['chunk_id']}: {exc}")
            continue
        key = normalize_question(question)
        if key in seen_questions:
            _log(f"skip  duplicate question: {question}")
            continue
        seen_questions.add(key)
        records.append(
            {
                "id": f"q-{uuid.uuid4().hex[:12]}",
                "query": question,
                "tenant_id": args.tenant,
                "relevant": [{"checksum": row["checksum"], "seq": int(row["seq"])}],
                "source_uri": _display_path(Path(str(row["source_uri"]))),
                "generator_model": _generator_label(settings),
                "generated_at": generated_at,
            }
        )
        _log(f"[{index}/{len(rows)}] {question}")

    if not records:
        raise SystemExit("no questions generated")
    _write_jsonl(args.out, records)
    _log(f"\nwrote {len(records)} questions to {args.out}")
    return 0


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


def _load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            refs = tuple(
                RelevanceRef(checksum=str(ref["checksum"]), seq=int(ref["seq"]))
                for ref in data["relevant"]
            )
            cases.append(
                EvalCase(
                    id=str(data["id"]),
                    query=str(data["query"]),
                    relevant=refs,
                    tenant_id=str(data.get("tenant_id", "")),
                    source_uri=str(data.get("source_uri", "")),
                    generator_model=str(data.get("generator_model", "")),
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"{path}:{number}: invalid query record: {exc}") from exc
    if not cases:
        raise SystemExit(f"{path} contains no queries")
    return cases


async def _corpus(
    engine: AsyncEngine, tenant: str
) -> tuple[dict[tuple[str, int], str], dict[str, int]]:
    async with engine.connect() as conn:
        refs = await _rows(
            conn,
            """
            SELECT c.id::text AS chunk_id, c.seq, d.checksum
              FROM chunks c
              JOIN documents d ON d.id = c.document_id
             WHERE c.tenant_id = :tenant
            """,
            {"tenant": tenant},
        )
        stats = await _one(
            conn,
            """
            SELECT count(*)::int AS chunks,
                   count(DISTINCT document_id)::int AS documents,
                   coalesce(max(vector_dims(embedding)), 0)::int AS dims
              FROM chunks
             WHERE tenant_id = :tenant
            """,
            {"tenant": tenant},
        )
    index = {(str(row["checksum"]), int(row["seq"])): str(row["chunk_id"]) for row in refs}
    return index, {
        "chunks": int(stats["chunks"]),
        "documents": int(stats["documents"]),
        "dims": int(stats["dims"]),
    }


async def _environment(engine: AsyncEngine) -> dict[str, str]:
    async with engine.connect() as conn:
        server = await _one(conn, "SELECT version() AS version")
        vector = await _rows(conn, "SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    return {
        "server": str(server["version"]),
        "pgvector": str(vector[0]["extversion"]) if vector else "unknown",
    }


def _table_name(run_id: str, arm: ArmSpec) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", arm.name.lower()).strip("_")
    return f"{EVAL_SCHEMA}.arm_{run_id}_{slug[:40]}"


@dataclass(frozen=True, slots=True)
class _PreparedArm:
    spec: ArmSpec
    sql: str | None
    table: str | None
    build_seconds: float
    retriever: Retriever | None = None


async def _materialize_existing(conn: AsyncConnection, table: str, tenant: str, dims: int) -> None:
    await conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
    await conn.execute(
        text(
            f"""
            CREATE TABLE {table} (
                chunk_id uuid PRIMARY KEY,
                document_id uuid NOT NULL,
                tenant_id varchar(64) NOT NULL,
                text text NOT NULL,
                embedding halfvec({dims}) NOT NULL
            )
            """
        )
    )
    await conn.execute(
        text(
            f"""
            INSERT INTO {table} (chunk_id, document_id, tenant_id, text, embedding)
            SELECT id,
                   document_id,
                   tenant_id,
                   text,
                   l2_normalize(subvector(embedding::vector, 1, {dims}))::halfvec({dims})
              FROM chunks
             WHERE tenant_id = :tenant
            """
        ),
        {"tenant": tenant},
    )


async def _materialize_reembed(
    conn: AsyncConnection,
    table: str,
    tenant: str,
    dims: int,
    model: str,
    settings: AtlasSettings,
    cache: _EmbeddingCache,
) -> None:
    embedder = NvidiaEmbedder(
        settings.nvidia.model_copy(update={"embedding_model": model, "embedding_dimensions": dims})
    )
    await conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
    await conn.execute(
        text(
            f"""
            CREATE TABLE {table} (
                chunk_id uuid PRIMARY KEY,
                document_id uuid NOT NULL,
                tenant_id varchar(64) NOT NULL,
                text text NOT NULL,
                embedding halfvec({dims}) NOT NULL
            )
            """
        )
    )
    rows = await _rows(
        conn,
        """
        SELECT id::text AS chunk_id, document_id::text AS document_id, tenant_id, text
          FROM chunks
         WHERE tenant_id = :tenant
         ORDER BY document_id, seq
        """,
        {"tenant": tenant},
    )
    batch_size = max(1, settings.nvidia.batch_size)
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        vectors = await _embed_cached(
            embedder,
            [str(row["text"]) for row in batch],
            EmbedKind.PASSAGE,
            f"re-embed {model}",
            model=model,
            dims=dims,
            cache=cache,
        )
        payload = [
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "tenant_id": row["tenant_id"],
                "text": row["text"],
                "embedding": _vector_literal(vector),
            }
            for row, vector in zip(batch, vectors, strict=True)
        ]
        await conn.execute(
            text(
                f"""
                INSERT INTO {table} (chunk_id, document_id, tenant_id, text, embedding)
                VALUES (:chunk_id, :document_id, :tenant_id, :text, CAST(:embedding AS halfvec({dims})))
                """
            ),
            payload,
        )
        _log(f"      re-embedded {min(start + batch_size, len(rows))}/{len(rows)} chunks")


async def _prepare_arm(
    engine: AsyncEngine,
    settings: AtlasSettings,
    tenant: str,
    arm: ArmSpec,
    run_id: str,
    corpus_dims: int,
    cache: _EmbeddingCache,
    vectors: VectorRepositoryPort,
    *,
    k: int,
) -> _PreparedArm:
    if arm.is_pipeline:
        retrieval = settings.retrieval.model_copy(
            update={
                "enable_rerank": arm.reranks,
                # Score the same depth the dense arms use; the configured value
                # (6 by default) is tuned for answer context, not evaluation.
                "final_top_k": max(k, settings.retrieval.final_top_k),
                "rerank_top_k": arm.rerank_top_k or settings.retrieval.rerank_top_k,
                # `dense-rerank` isolates the lexical branch's contribution:
                # a zero limit makes `lexical_search` return nothing, so RRF
                # fuses a single list without changing production code.
                "lexical_top_k": 0
                if arm.kind == "dense-rerank"
                else settings.retrieval.lexical_top_k,
            }
        )
        embedder = _CachedEmbedder(
            NvidiaEmbedder(
                settings.nvidia.model_copy(update={"embedding_dimensions": corpus_dims})
            ),
            model=settings.nvidia.embedding_model,
            dims=corpus_dims,
            cache=cache,
        )
        reranker: RerankerPort | None = NvidiaReranker(settings.nvidia) if arm.reranks else None
        return _PreparedArm(
            spec=arm,
            sql=None,
            table=None,
            build_seconds=0.0,
            retriever=Retriever(
                settings=retrieval,
                vectors=vectors,
                embedder=embedder,
                reranker=reranker,
            ),
        )

    started = time.perf_counter()
    table: str | None = None
    needs_table = arm.kind == "reembed" or arm.ann
    if needs_table:
        table = _table_name(run_id, arm)
        try:
            async with engine.begin() as conn:
                await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {EVAL_SCHEMA}"))
                if arm.kind == "existing":
                    await _materialize_existing(conn, table, tenant, arm.dims)
                else:
                    assert arm.model is not None
                    await _materialize_reembed(
                        conn, table, tenant, arm.dims, arm.model, settings, cache
                    )
                await conn.execute(text(f"ANALYZE {table}"))
                if arm.ann:
                    await conn.execute(text("SET maintenance_work_mem = '1GB'"))
                    # Docker /dev/shm is 64 MB; parallel HNSW builds fail there.
                    await conn.execute(text("SET max_parallel_maintenance_workers = 0"))
                    await conn.execute(
                        text(
                            f"CREATE INDEX ON {table} USING hnsw (embedding halfvec_cosine_ops) "
                            "WITH (m = 16, ef_construction = 64)"
                        )
                    )
        except Exception:
            async with engine.begin() as cleanup:
                await cleanup.execute(text(f"DROP TABLE IF EXISTS {table}"))
            raise
    build_seconds = time.perf_counter() - started

    if table is None:
        if arm.dims == corpus_dims:
            sql = (
                "SELECT id::text AS chunk_id FROM chunks WHERE tenant_id = :tenant "
                f"ORDER BY embedding <=> CAST(:vec AS halfvec({arm.dims})) LIMIT :k"
            )
        else:
            sql = (
                "SELECT id::text AS chunk_id FROM chunks WHERE tenant_id = :tenant "
                f"ORDER BY l2_normalize(subvector(embedding::vector, 1, {arm.dims}))"
                f"::halfvec({arm.dims}) <=> CAST(:vec AS halfvec({arm.dims})) LIMIT :k"
            )
    else:
        sql = (
            f"SELECT chunk_id::text AS chunk_id FROM {table} WHERE tenant_id = :tenant "
            f"ORDER BY embedding <=> CAST(:vec AS halfvec({arm.dims})) LIMIT :k"
        )
    return _PreparedArm(spec=arm, sql=sql, table=table, build_seconds=build_seconds)


async def _search(
    conn: AsyncConnection, sql: str, tenant: str, vector: Sequence[float], k: int
) -> tuple[list[str], float]:
    started = time.perf_counter()
    result = await conn.execute(
        text(sql), {"tenant": tenant, "vec": _vector_literal(vector), "k": k}
    )
    identifiers = [str(row[0]) for row in result.all()]
    return identifiers, (time.perf_counter() - started) * 1000.0


async def _query_vectors(
    settings: AtlasSettings,
    cases: Sequence[EvalCase],
    corpus_dims: int,
    cache: _EmbeddingCache,
) -> list[list[float]]:
    # The corpus column, not the current config, decides the query width: an
    # evaluation must embed queries with the same model that produced the
    # stored passages.
    embedder = NvidiaEmbedder(
        settings.nvidia.model_copy(update={"embedding_dimensions": corpus_dims})
    )
    _log(
        f"embedding {len(cases)} queries with {settings.nvidia.embedding_model} at {corpus_dims} dims"
    )
    return await _embed_cached(
        embedder,
        [case.query for case in cases],
        EmbedKind.QUERY,
        "query embedding",
        model=settings.nvidia.embedding_model,
        dims=corpus_dims,
        cache=cache,
    )


async def _reembed_query_vectors(
    settings: AtlasSettings, cases: Sequence[EvalCase], arm: ArmSpec, cache: _EmbeddingCache
) -> list[list[float]]:
    assert arm.model is not None
    embedder = NvidiaEmbedder(
        settings.nvidia.model_copy(
            update={"embedding_model": arm.model, "embedding_dimensions": arm.dims}
        )
    )
    _log(f"embedding {len(cases)} queries with {arm.model} at {arm.dims} dims")
    return await _embed_cached(
        embedder,
        [case.query for case in cases],
        EmbedKind.QUERY,
        f"query embedding ({arm.model})",
        model=arm.model,
        dims=arm.dims,
        cache=cache,
    )


async def _warm_pipeline_embeddings(
    settings: AtlasSettings,
    cases: Sequence[EvalCase],
    corpus_dims: int,
    cache: _EmbeddingCache,
) -> None:
    """Populate the cache for every sub-query the retriever will embed.

    `Retriever` splits compound questions before embedding, so full-query cache
    entries are not enough. Without this pass the first pipeline arm would pay
    cold NIM latency and report it as retrieval latency.
    """
    sub_queries = sorted({sub for case in cases for sub in split_query(case.query)})
    embedder = NvidiaEmbedder(
        settings.nvidia.model_copy(update={"embedding_dimensions": corpus_dims})
    )
    _log(f"warming {len(sub_queries)} sub-query embeddings for pipeline arms")
    await _embed_cached(
        embedder,
        sub_queries,
        EmbedKind.QUERY,
        "pipeline warm-up",
        model=settings.nvidia.embedding_model,
        dims=corpus_dims,
        cache=cache,
    )


@dataclass(slots=True)
class _ArmOutcome:
    spec: ArmSpec
    build_seconds: float
    summary: ScoreSummary
    per_case: list[tuple[EvalCase, QueryScore]]
    latencies: list[float]
    reranked: int | None = None


async def _evaluate_pipeline_arm(
    retriever: Retriever,
    prepared: _PreparedArm,
    tenant: str,
    cases: Sequence[EvalCase],
    relevant: Sequence[Sequence[str]],
    k: int,
) -> _ArmOutcome:
    """Score the production retriever, latency included (embed + SQL + rerank)."""
    scores: list[QueryScore] = []
    per_case: list[tuple[EvalCase, QueryScore]] = []
    latencies: list[float] = []
    reranked = 0
    for case, positive_ids in zip(cases, relevant, strict=True):
        started = time.perf_counter()
        outcome = await retriever.retrieve(
            Query(id=uuid.uuid4(), tenant_id=tenant, text=case.query)
        )
        latencies.append((time.perf_counter() - started) * 1000.0)
        if any(step.name == "rerank" and "applied" in step.detail for step in outcome.trace):
            reranked += 1
        ranked = [str(context.chunk_id) for context in outcome.contexts]
        score = score_ranking(ranked, positive_ids, k=k)
        scores.append(score)
        per_case.append((case, score))
    return _ArmOutcome(
        spec=prepared.spec,
        build_seconds=prepared.build_seconds,
        summary=summarize(scores, latencies),
        per_case=per_case,
        latencies=latencies,
        # `hybrid` intentionally does not rerank; only report coverage for arms
        # that claim to.
        reranked=reranked if prepared.spec.reranks else None,
    )


async def _evaluate_arm(
    engine: AsyncEngine,
    prepared: _PreparedArm,
    tenant: str,
    cases: Sequence[EvalCase],
    relevant: Sequence[Sequence[str]],
    query_vectors: Sequence[Sequence[float]],
    k: int,
    ef_search: int,
) -> _ArmOutcome:
    if prepared.retriever is not None:
        return await _evaluate_pipeline_arm(
            prepared.retriever, prepared, tenant, cases, relevant, k
        )
    assert prepared.sql is not None
    scores: list[QueryScore] = []
    per_case: list[tuple[EvalCase, QueryScore]] = []
    latencies: list[float] = []
    async with engine.connect() as conn:
        if prepared.spec.ann:
            # On small corpora the planner prefers a seq scan; force the index so
            # this arm measures HNSW rather than an accidental exact scan.
            await conn.execute(text("SET enable_seqscan = off"))
            await conn.execute(
                text("SELECT set_config('hnsw.ef_search', :value, false)"),
                {"value": str(ef_search)},
            )
        for case, positive_ids, vector in zip(cases, relevant, query_vectors, strict=True):
            ranked, elapsed = await _search(conn, prepared.sql, tenant, vector, k)
            score = score_ranking(ranked, positive_ids, k=k)
            scores.append(score)
            per_case.append((case, score))
            latencies.append(elapsed)
    return _ArmOutcome(
        spec=prepared.spec,
        build_seconds=prepared.build_seconds,
        summary=summarize(scores, latencies),
        per_case=per_case,
        latencies=latencies,
    )


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _delta_row(name: str, baseline: Sequence[float], arm: Sequence[float]) -> str:
    delta = bootstrap_delta(baseline, arm)
    return f"| {name} | {delta.mean * 100:+.1f} pp | {delta.low * 100:+.1f} … {delta.high * 100:+.1f} pp |"


def _markdown(
    *,
    tenant: str,
    k: int,
    cases: Sequence[EvalCase],
    skipped: int,
    corpus: Mapping[str, int],
    environment: Mapping[str, str],
    settings: AtlasSettings,
    outcomes: Sequence[_ArmOutcome],
    baseline: _ArmOutcome,
    queries_path: Path,
    label: str,
) -> str:
    generated = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    models = sorted({outcome.spec.model or settings.nvidia.embedding_model for outcome in outcomes})
    pipeline = [outcome for outcome in outcomes if outcome.spec.is_pipeline]
    reranked = [outcome for outcome in outcomes if outcome.reranked is not None]
    title = "# Retrieval evaluation — Atlas corpus"
    if label:
        title = f"{title} ({label})"
    scoring = (
        "Dense arms score cosine distance only; `hybrid` arms run production RRF"
        if pipeline
        else "Dense-only: cosine distance, no fusion or reranking"
    )
    lines = [
        title,
        "",
        f"Generated {generated}.",
        "",
        "## Setup",
        "",
        f"- Tenant: `{tenant}`",
        f"- Corpus: **{corpus['chunks']:,} chunks** from {corpus['documents']:,} documents, "
        f"{corpus['dims']} dims",
        f"- Queries: **{len(cases)}** scored, {skipped} skipped (ground truth no longer resolvable)",
        f"- Query set: `{queries_path}`",
        f"- Query embedder(s): {', '.join(f'`{model}`' for model in models)}",
        f"- Top-{k}; {scoring}",
    ]
    if reranked:
        coverage = ", ".join(
            f"`{outcome.spec.name}` {outcome.reranked}/{outcome.summary.count}"
            for outcome in reranked
        )
        lines.append(f"- Rerank coverage: {coverage} queries")
    lines += [
        f"- Server: `{environment['server']}`; pgvector `{environment['pgvector']}`",
        "",
        "## Results",
        "",
        f"| arm | recall@{k} | MRR | nDCG@{k} | hit@1 | p50 | p95 | build |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for outcome in outcomes:
        summary = outcome.summary
        lines.append(
            f"| `{outcome.spec.name}` | {_pct(summary.recall)} | {summary.mrr:.3f} | "
            f"{_pct(summary.ndcg)} | {_pct(summary.hit_at_1)} | "
            f"{summary.p50_ms:.1f} ms | {summary.p95_ms:.1f} ms | "
            f"{outcome.build_seconds:.1f} s |"
        )

    lines += [
        "",
        f"## Difference vs `{baseline.spec.name}`",
        "",
        "Paired bootstrap over queries (2,000 resamples, 95% interval). An interval",
        "that includes zero is not distinguishable from query-sampling noise.",
        "",
        "| arm | Δ recall | Δ nDCG |",
        "| --- | ---: | ---: |",
    ]
    for outcome in outcomes:
        if outcome.spec.name == baseline.spec.name:
            continue
        lines.append(
            _delta_row(
                f"`{outcome.spec.name}` recall",
                [score.recall for _, score in baseline.per_case],
                [score.recall for _, score in outcome.per_case],
            )
        )
        lines.append(
            _delta_row(
                f"`{outcome.spec.name}` nDCG",
                [score.ndcg for _, score in baseline.per_case],
                [score.ndcg for _, score in outcome.per_case],
            )
        )

    regressions: list[tuple[str, str]] = []
    baseline_scores = {case.id: score for case, score in baseline.per_case}
    for outcome in outcomes:
        if outcome.spec.name == baseline.spec.name:
            continue
        for case, score in outcome.per_case:
            base = baseline_scores[case.id]
            if base.rank is None:
                continue
            # Ignore one-rank shuffles: they are common noise on a small set.
            if score.rank is None or score.rank - base.rank >= 3:
                regressions.append(
                    (
                        outcome.spec.name,
                        f"{case.query} (rank {base.rank} → "
                        f"{score.rank if score.rank is not None else 'miss'})",
                    )
                )
    if regressions:
        lines += ["", "## Material regressions (missed, or dropped 3+ ranks)", ""]
        for arm_name, description in regressions[:10]:
            lines.append(f"- `{arm_name}`: {description}")
        if len(regressions) > 10:
            lines.append(f"- …and {len(regressions) - 10} more in the JSON report")

    lines += [
        "",
        "## Caveats",
        "",
        "- Questions are generated from the source chunk, so they reuse its",
        "  vocabulary. Real user questions drift further from the passage, which",
        "  makes this an optimistic estimate of absolute quality.",
        "- Ground truth is the single source chunk. Overlapping chunks or passages",
        "  that also answer the question count as misses, which understates recall.",
        "- Corpus size bounds how far these numbers extrapolate; recall is easier on",
        "  small corpora. Use the deltas between arms, not the absolute values.",
        "- ANN arms force `enable_seqscan = off` so the HNSW index is actually used.",
        "  Their latency is not directly comparable to exact arms on tiny corpora.",
        "- Dense-arm latency measures the SQL query only. `hybrid` arms also include",
        "  query embedding (cached here, so near zero) and, for reranked arms, the",
        "  hosted rerank round trip. That is the number users feel, but it is not a",
        "  like-for-like comparison with dense SQL.",
        "",
    ]
    return "\n".join(lines)


async def cmd_warm(args: argparse.Namespace) -> int:
    """Cache every query vector a run needs, retrying a cold embedder.

    The hosted 8B embedder returns 504s for minutes at a time. Warming is
    separate from scoring so a slow endpoint cannot produce a half-written
    report: retry this in a loop, then `run` against a warm cache.
    """
    settings = _settings(args.url)
    cases = _load_cases(args.queries)
    if not cases:
        raise SystemExit(f"{args.queries} contains no queries")

    engine, _ = create_engine_and_session(settings.postgres)
    try:
        _, corpus = await _corpus(engine, args.tenant)
    finally:
        await engine.dispose()
    if corpus["chunks"] == 0:
        raise SystemExit(f"tenant {args.tenant!r} has no chunks; run `ingest` first")
    dims = corpus["dims"]

    texts = sorted(
        {case.query for case in cases} | {sub for case in cases for sub in split_query(case.query)}
    )
    model = settings.nvidia.embedding_model
    cache = _EmbeddingCache(args.cache)
    missing = [
        text for text in texts if cache.get(_embed_key(model, dims, EmbedKind.QUERY, text)) is None
    ]
    if not missing:
        _log(f"cache already warm: {len(texts)} query vectors at {dims} dims")
        return 0
    _log(f"{len(missing)}/{len(texts)} query vectors missing at {dims} dims")

    embedder = NvidiaEmbedder(settings.nvidia.model_copy(update={"embedding_dimensions": dims}))
    for attempt in range(1, args.attempts + 1):
        try:
            vectors = await embedder.embed(missing, EmbedKind.QUERY)
        except InfraError as exc:
            if attempt == args.attempts:
                _log(f"warning  embedder still failing after {attempt} attempt(s): {exc}")
                return 1
            wait = min(args.wait * attempt, args.max_wait)
            _log(f"cold start (attempt {attempt}); waiting {wait:.0f}s")
            await asyncio.sleep(wait)
            continue
        if len(vectors) != len(missing):
            _log(f"warning  expected {len(missing)} vectors, received {len(vectors)}")
            return 1
        cache.put_many(
            (_embed_key(model, dims, EmbedKind.QUERY, text), vector)
            for text, vector in zip(missing, vectors, strict=True)
        )
        _log(f"cached {len(missing)} query vectors at {dims} dims")
        return 0
    return 1


async def cmd_run(args: argparse.Namespace) -> int:
    settings = _settings(args.url)
    arms = parse_arms(args.arms)
    cases = _load_cases(args.queries)
    engine, session_factory = create_engine_and_session(settings.postgres)
    run_id = f"{datetime.datetime.now(datetime.UTC):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}"
    tables: list[str] = []
    try:
        environment = await _environment(engine)
        index, corpus = await _corpus(engine, args.tenant)
        if corpus["chunks"] == 0:
            raise SystemExit(f"tenant {args.tenant!r} has no chunks; run `ingest` first")
        for arm in arms:
            if arm.kind == "existing" and arm.dims > corpus["dims"]:
                raise SystemExit(
                    f"arm {arm.name!r} asks for {arm.dims} dims but the corpus stores "
                    f"{corpus['dims']}; use a reembed arm instead"
                )

        resolved: list[EvalCase] = []
        relevant: list[tuple[str, ...]] = []
        skipped = 0
        for case in cases:
            positive_ids, missing = resolve_case(case, index)
            if missing:
                _log(f"skip  {case.id}: {missing} ground-truth reference(s) missing")
                skipped += 1
                continue
            resolved.append(case)
            relevant.append(positive_ids)
        if not resolved:
            raise SystemExit("no query has resolvable ground truth")

        cache = _EmbeddingCache(args.cache)
        base_vectors: list[list[float]] = []
        if any(arm.kind == "existing" for arm in arms):
            base_vectors = await _query_vectors(settings, resolved, corpus["dims"], cache)
        if any(arm.is_pipeline for arm in arms):
            await _warm_pipeline_embeddings(settings, resolved, corpus["dims"], cache)
        vectors_by_arm: dict[str, list[list[float]]] = {}
        for arm in arms:
            if arm.kind == "existing":
                vectors_by_arm[arm.name] = [
                    base if arm.dims == corpus["dims"] else truncate_embedding(base, arm.dims)
                    for base in base_vectors
                ]
            elif arm.kind == "reembed":
                assert arm.model is not None
                vectors_by_arm[arm.name] = await _reembed_query_vectors(
                    settings, resolved, arm, cache
                )
            else:
                # Pipeline arms embed each sub-query through `_CachedEmbedder`.
                vectors_by_arm[arm.name] = []

        outcomes: list[_ArmOutcome] = []
        for arm in arms:
            _log(f"preparing {arm.name} ...")
            prepared = await _prepare_arm(
                engine,
                settings,
                args.tenant,
                arm,
                run_id,
                corpus["dims"],
                cache,
                PgVectorRepository(session_factory),
                k=args.k,
            )
            if prepared.table is not None:
                tables.append(prepared.table)
            _log(
                f"evaluating {arm.name} "
                f"(build {prepared.build_seconds:.1f}s, {len(resolved)} queries) ..."
            )
            outcome = await _evaluate_arm(
                engine,
                prepared,
                args.tenant,
                resolved,
                relevant,
                vectors_by_arm[arm.name],
                args.k,
                args.ef_search,
            )
            outcomes.append(outcome)
            summary = outcome.summary
            if arm.reranks:
                reranked = outcome.reranked or 0
                if reranked == 0:
                    raise SystemExit(
                        f"{arm.name} never applied reranking (provider unavailable?); "
                        "refusing to report a no-rerank result as reranked"
                    )
                if reranked < summary.count:
                    _log(
                        f"  warning: rerank applied to {reranked}/{summary.count} "
                        "queries; the rest were skipped after a provider failure"
                    )
            _log(
                f"  recall@{args.k}={_pct(summary.recall)} ndcg@{args.k}={_pct(summary.ndcg)} "
                f"p50={summary.p50_ms:.1f}ms p95={summary.p95_ms:.1f}ms"
            )

        baseline = outcomes[0]
        payload = {
            "generated_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "environment": dict(environment),
            "config": {
                "tenant": args.tenant,
                "k": args.k,
                "ef_search": args.ef_search,
                "queries": str(args.queries),
                "query_count": len(resolved),
                "skipped": skipped,
                "query_embedders": sorted(
                    {arm.model or settings.nvidia.embedding_model for arm in arms}
                ),
                "arms": [arm.name for arm in arms],
            },
            "corpus": dict(corpus),
            "results": [
                {
                    "arm": outcome.spec.name,
                    "kind": outcome.spec.kind,
                    "dims": outcome.spec.dims,
                    "ann": outcome.spec.ann,
                    "model": outcome.spec.model,
                    "build_seconds": round(outcome.build_seconds, 2),
                    "recall": outcome.summary.recall,
                    "mrr": outcome.summary.mrr,
                    "ndcg": outcome.summary.ndcg,
                    "hit_at_1": outcome.summary.hit_at_1,
                    "p50_ms": outcome.summary.p50_ms,
                    "p95_ms": outcome.summary.p95_ms,
                    "reranked": outcome.reranked,
                    "delta_vs_baseline": {
                        "recall": asdict(
                            bootstrap_delta(
                                [score.recall for _, score in baseline.per_case],
                                [score.recall for _, score in outcome.per_case],
                            )
                        ),
                        "ndcg": asdict(
                            bootstrap_delta(
                                [score.ndcg for _, score in baseline.per_case],
                                [score.ndcg for _, score in outcome.per_case],
                            )
                        ),
                    },
                    "ranks": [
                        {"query_id": case.id, "rank": score.rank, "ndcg": score.ndcg}
                        for case, score in outcome.per_case
                    ],
                }
                for outcome in outcomes
            ],
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"-{args.label}" if args.label else ""
        json_path = args.out_dir / f"eval-retrieval{suffix}.json"
        markdown_path = args.out_dir / f"EVAL-retrieval{suffix}.md"
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        markdown_path.write_text(
            _markdown(
                tenant=args.tenant,
                k=args.k,
                cases=resolved,
                skipped=skipped,
                corpus=corpus,
                environment=environment,
                settings=settings,
                outcomes=outcomes,
                baseline=baseline,
                queries_path=args.queries,
                label=args.label,
            )
            + "\n",
            encoding="utf-8",
        )
        _log(f"\nwrote {markdown_path} and {json_path.name}")
    finally:
        if tables and not args.keep_tables:
            async with engine.begin() as conn:
                for table in tables:
                    await conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
        await engine.dispose()
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--url",
        default=None,
        help="Postgres URL; defaults to POSTGRES__URL or the Atlas default",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", parents=[common], help="ingest a corpus into a tenant")
    ingest.add_argument("--path", action="append", required=True, help="file or directory")
    ingest.add_argument("--tenant", default=DEFAULT_TENANT)
    ingest.add_argument("--extensions", default=".md,.markdown,.txt,.html,.htm,.pdf")
    ingest.set_defaults(func=cmd_ingest)

    generate = sub.add_parser("generate", parents=[common], help="generate judged queries")
    generate.add_argument("--tenant", default=DEFAULT_TENANT)
    generate.add_argument("--sample", type=int, default=40)
    generate.add_argument("--seed", type=int, default=7)
    generate.add_argument("--min-chars", type=int, default=200)
    generate.add_argument("--max-chars", type=int, default=6000)
    generate.add_argument("--out", type=Path, default=Path("specs/eval/queries-atlas-repo.jsonl"))
    generate.add_argument("--overwrite", action="store_true")
    generate.set_defaults(func=cmd_generate)

    run = sub.add_parser("run", parents=[common], help="evaluate retrieval arms")
    run.add_argument("--tenant", default=DEFAULT_TENANT)
    run.add_argument("--queries", type=Path, required=True)
    run.add_argument("--arms", default=DEFAULT_ARMS)
    run.add_argument("--k", type=int, default=10)
    run.add_argument("--ef-search", type=int, default=DEFAULT_EF_SEARCH)
    run.add_argument("--out-dir", type=Path, default=Path("specs/verifications"))
    run.add_argument("--label", default="", help="suffix for report filenames and title")
    run.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    run.add_argument("--keep-tables", action="store_true")
    run.set_defaults(func=cmd_run)

    warm = sub.add_parser("warm", parents=[common], help="cache query embeddings ahead of a run")
    warm.add_argument("--tenant", default=DEFAULT_TENANT)
    warm.add_argument("--queries", type=Path, required=True)
    warm.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    warm.add_argument("--attempts", type=int, default=10)
    warm.add_argument("--wait", type=float, default=120.0)
    warm.add_argument("--max-wait", type=float, default=900.0)
    warm.set_defaults(func=cmd_warm)

    return parser


def main() -> int:
    args = _parser().parse_args()
    return int(asyncio.run(args.func(args)))


if __name__ == "__main__":
    raise SystemExit(main())
