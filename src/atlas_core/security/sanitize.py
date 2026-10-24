"""Input/answer sanitization: normalization, length caps, control-char stripping."""

from __future__ import annotations

import html
import unicodedata

_CONTROL_CHARS = "".join(chr(c) for c in range(0x00, 0x20) if c not in (0x09, 0x0A, 0x0C, 0x0D))
_MAX_CHARS = 200_000


class Sanitizer:
    """Pure text hygiene used on ingest payloads and retrieved context."""

    def clean(self, text: str, *, strip_html: bool = False) -> str:
        if strip_html:
            text = html.unescape(text)
            text = "".join(ch for ch in text if ch not in "<>")
        cleaned = unicodedata.normalize("NFKC", text)
        cleaned = "".join(ch for ch in cleaned if ch not in _CONTROL_CHARS)
        cleaned = cleaned.replace("\x00", "").replace("\ufeff", "")
        return cleaned[:_MAX_CHARS]

    @staticmethod
    def enforce_limit(text: str, limit: int) -> str:
        return text[:limit]
