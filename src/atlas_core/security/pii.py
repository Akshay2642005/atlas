"""Regex-based PII detection + masking.

Swappable via `PiiDetectorPort`; a Presidio-based adapter (NER-grade detection)
can be dropped in later without touching call sites.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from atlas_core.domain.ports import PiiDetectorPort, PiiFinding


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_iban = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
_CC = re.compile(r"\b(?:\d[ -]?){13,19}\b")

@dataclass(frozen=True, slots=True)
class RegexPiiDetector(PiiDetectorPort):
    """Deterministic detector covering the highest-risk PII categories."""

    def detect(self, text: str) -> list[PiiFinding]:
        findings: list[PiiFinding] = []
        for kind, pattern in self._rules():
            for match in pattern.finditer(text):
                kind_final = kind
                if kind == "credit_card" and not _luhn_ok(re.sub(r"[ -]", "", match.group())):
                    continue
                findings.append(
                    PiiFinding(kind=kind_final, start=match.start(), end=match.end(), value=match.group())
                )
        return findings

    def _rules(self) -> tuple[tuple[str, re.Pattern[str]], ...]:
        return (
            ("email", _EMAIL),
            ("ssn", _SSN),
            ("credit_card", _CC),
            ("iban", _iban),
            ("phone", _PHONE),
        )


_MAX_PLACEHOLDER: Final = "[{kind}]"


def mask_text(text: str, findings: list[PiiFinding], *, placeholder: str = _MAX_PLACEHOLDER) -> str:
    """Mask from right to left so earlier spans stay valid."""
    for finding in sorted(findings, key=lambda f: f.start, reverse=True):
        text = text[: finding.start] + placeholder.format(kind=finding.kind.upper()) + text[finding.end :]
    return text
