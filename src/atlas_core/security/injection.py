"""Prompt-injection guard (heuristic, direction: inbound content and outbound answers)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from atlas_core.domain.ports import InjectionGuardPort, InjectionVerdict

_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (?:all |any )?(?:previous|prior|above) (?:instructions|prompts)",
        r"disregard (?:all |any )?(?:previous|prior|above)",
        r"reveal (?:your|the) (?:system )?(?:prompt|instructions)",
        r"you are now (?:a|an|in)",
        r"jailbreak",
        r"developer mode",
        r"print (?:your|the) (?:system )?prompt",
        r"what (?:are )?your (?:instructions|system prompt)",
        r"system\s*:\s*",
        r"<\|im_start\|>"
    )
)


@dataclass(frozen=True, slots=True)
class HeuristicInjectionGuard(InjectionGuardPort):
    def scan(self, text: str) -> InjectionVerdict:
        matches = tuple(m.group(0) for m in (p.search(text) for p in _PATTERNS) if m)
        return InjectionVerdict(risky=bool(matches), matches=matches)
