"""Build an exact-identifier query set from an ingested tenant.

The question-shaped query sets are generated from the target chunk, so they
reuse its vocabulary and dense retrieval already does well. That leaves the
lexical branch's actual justification untested: users typing an exact
identifier (`MultipleObjectMixin`, `ALLOWED_HOSTS`, `default_app_config`),
where an embedding has nothing to generalise from.

This samples chunks, extracts an identifier that is rare in the corpus but
present in the chunk, and emits the identifier *alone* as the query with that
chunk as ground truth. Rarity keeps the query discriminative: an identifier
that appears in 200 chunks has no single right answer.

    uv run python scripts/gen_identifier_queries.py --tenant atlas-eval-docs \\
        --out specs/eval/queries-public-docs-identifiers.jsonl

Like the other query sets it stores ground truth as `(checksum, chunk seq)`, so
it survives re-ingestion.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from atlas_core.config import load_settings
from atlas_vector.engine import create_engine_and_session

# snake_case (>=2 parts) or CamelCase with an internal capital.
_IDENTIFIER_RE = re.compile(
    r"\b(?:[a-z][a-z0-9]*(?:_[a-z0-9]+)+|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)\b"
)
# Words that are identifiers syntactically but carry no retrieval signal.
_STOP_IDENTIFIERS = frozenset(
    {
        "utf_8",
        "iso_8859_1",
        "text_html",
        "application_json",
        "content_type",
        "class_based",
    }
)


def _candidates(text_value: str) -> list[str]:
    """Identifiers long enough to be unambiguous and not test fixtures."""
    return sorted(
        {
            match
            for match in _IDENTIFIER_RE.findall(text_value)
            if len(match) >= 10
            and match.lower() not in _STOP_IDENTIFIERS
            and not match.startswith("test_")
        }
    )


async def _chunks(engine: AsyncEngine, tenant: str) -> list[dict[str, object]]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                """
                SELECT c.id::text AS chunk_id, v.checksum, c.seq, c.text
                  FROM chunks c
                  JOIN documents d ON d.id = c.document_id
                  JOIN document_versions v ON v.id = c.document_version_id
                 WHERE c.tenant_id = :tenant
                 ORDER BY v.checksum, c.seq
                """
            ),
            {"tenant": tenant},
        )
        return [
            {"chunk_id": row[0], "checksum": row[1], "seq": row[2], "text": row[3]} for row in rows
        ]


def _pick(
    chunks: list[dict[str, object]], *, sample: int, max_df: int, seed: int
) -> list[dict[str, object]]:
    """Choose one rare identifier per sampled chunk."""
    frequency: Counter[str] = Counter()
    holders: dict[str, set[str]] = defaultdict(set)
    per_chunk: dict[str, list[str]] = {}
    for chunk in chunks:
        chunk_id = str(chunk["chunk_id"])
        found = _candidates(str(chunk["text"]))
        per_chunk[chunk_id] = found
        for identifier in found:
            frequency[identifier] += 1
            holders[identifier].add(chunk_id)

    rng = random.Random(seed)
    order = sorted(per_chunk)
    rng.shuffle(order)
    picked: list[dict[str, object]] = []
    by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    for chunk_id in order:
        rare = [i for i in per_chunk[chunk_id] if frequency[i] <= max_df]
        if not rare:
            continue
        identifier = rng.choice(rare)
        chunk = by_id[chunk_id]
        picked.append(
            {
                "id": f"ident-{len(picked) + 1:03d}",
                "query": identifier,
                "relevant": [{"checksum": chunk["checksum"], "seq": chunk["seq"]}],
                "source_uri": f"df={frequency[identifier]}",
            }
        )
        if len(picked) == sample:
            break
    return picked


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", default="atlas-eval-docs")
    parser.add_argument(
        "--out", type=Path, default=Path("specs/eval/queries-public-docs-identifiers.jsonl")
    )
    parser.add_argument("--sample", type=int, default=60)
    parser.add_argument(
        "--max-df", type=int, default=3, help="max chunks an identifier may appear in"
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    settings = load_settings()
    engine, _ = create_engine_and_session(settings.postgres)
    try:
        chunks = await _chunks(engine, args.tenant)
        if not chunks:
            raise SystemExit(f"tenant {args.tenant!r} has no chunks")
        picked = _pick(chunks, sample=args.sample, max_df=args.max_df, seed=args.seed)
        if not picked:
            raise SystemExit("no rare identifiers found; raise --max-df")
        now = datetime.now(UTC).isoformat()
        rows = [
            {
                **item,
                "tenant_id": args.tenant,
                "generator_model": "identifier-extraction",
                "generated_at": now,
            }
            for item in picked
        ]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        print(f"wrote {len(rows)} identifier queries to {args.out}")
    finally:
        await engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
