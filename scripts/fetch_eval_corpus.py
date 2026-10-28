"""Fetch a pinned, public documentation corpus for the retrieval evaluation.

The repository corpus is 112 chunks: enough to catch gross regressions, too
small to separate 4096 from 2048 or to contain the exact-identifier queries
where lexical retrieval should matter. This downloads a few public
documentation trees at pinned tags so the width and fusion questions can be
asked at ~10x the scale, reproducibly, without sending private text to a hosted
embedder.

    uv run python scripts/fetch_eval_corpus.py
    POSTGRES__URL=... uv run python scripts/eval_retrieval.py ingest \\
        --tenant atlas-eval-docs --path .eval-corpora/fastapi ...

Corpus files are local state (`.eval-corpora/`, gitignored); the pinned refs in
this file are the reproducibility contract.
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path(".eval-corpora")


@dataclass(frozen=True, slots=True)
class Source:
    """One documentation tree at an immutable tag."""

    name: str
    repo: str
    ref: str
    subpath: str
    suffixes: tuple[str, ...]


SOURCES: tuple[Source, ...] = (
    Source("fastapi", "fastapi/fastapi", "0.162.0", "docs/en/docs", (".md",)),
    Source("django", "django/django", "6.1.2", "docs", (".txt",)),
    Source("pydantic", "pydantic/pydantic", "v2.14.1", "docs", (".md",)),
    Source("pgvector", "pgvector/pgvector", "v0.8.7", "", (".md",)),
)


def _tarball_url(source: Source) -> str:
    return f"https://codeload.github.com/{source.repo}/tar.gz/refs/tags/{source.ref}"


def _matches(source: Source, relative: str) -> bool:
    if not relative.endswith(source.suffixes):
        return False
    prefix = source.subpath.rstrip("/")
    return not prefix or relative.startswith(f"{prefix}/")


def _extract(source: Source, destination: Path) -> tuple[int, int]:
    """Download the tag and write the matching docs under `destination`."""
    files = 0
    total_bytes = 0
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as handle:
        with urllib.request.urlopen(_tarball_url(source), timeout=180) as response:
            shutil.copyfileobj(response, handle)
        handle.flush()
        with tarfile.open(handle.name) as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                # Drop the archive's single top-level directory.
                _, _, relative = member.name.partition("/")
                if not relative or not _matches(source, relative):
                    continue
                payload = archive.extractfile(member)
                if payload is None:
                    continue
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload.read())
                files += 1
                total_bytes += member.size
    return files, total_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--force", action="store_true", help="refetch existing trees")
    args = parser.parse_args()

    for source in SOURCES:
        destination = args.root / source.name
        if destination.exists() and any(destination.rglob("*")) and not args.force:
            existing = sum(1 for path in destination.rglob("*") if path.is_file())
            print(f"{source.name:10s} already fetched ({existing} files)")
            continue
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        files, size = _extract(source, destination)
        if files == 0:
            raise SystemExit(f"{source.name}: no files matched {source.subpath!r}")
        (destination / "SOURCE.txt").write_text(
            f"{source.repo}@{source.ref}\nsubpath={source.subpath or '.'}\nfiles={files}\n",
            encoding="utf-8",
        )
        print(
            f"{source.name:10s} {files:4d} files, {size / 1_000_000:.1f} MB  ({source.repo}@{source.ref})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
