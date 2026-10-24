"""Loaders: turn raw bytes into plain text. Pure, side-effect free."""

from __future__ import annotations

import io
import re

import pypdf

from atlas_core.domain.errors import IngestError

MIME_PLAIN = "text/plain"
MIME_MARKDOWN = "text/markdown"
MIME_HTML = "text/html"
MIME_PDF = "application/pdf"
SUPPORTED = frozenset({MIME_PLAIN, MIME_MARKDOWN, MIME_HTML, MIME_PDF})

_TAG_RE = re.compile(r"<[^>]+>")


def _pdf_to_text(data: bytes) -> str:
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001 — pypdf raises many exceptions
        raise IngestError(f"failed to parse PDF: {exc}") from exc
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def load_bytes(data: bytes, mime: str) -> str:
    """Normalize arbitrary source bytes into plain text."""
    if mime not in SUPPORTED:
        raise IngestError(f"unsupported mime: {mime}", details={"supported": sorted(SUPPORTED)})
    if mime == MIME_PDF:
        return _pdf_to_text(data)
    text = data.decode("utf-8", errors="replace")
    return _TAG_RE.sub(" ", text) if mime == MIME_HTML else text


def checksum(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()
