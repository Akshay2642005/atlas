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
        # A chat role prefix is a *block* that starts with `system:`. Line-anchoring
        # alone is not enough: wrapped prose puts the word at a line start too
        # ("Django uses an intuitive\nsystem: A model class …"), which rejected
        # real documentation (BUG-007). Roles are separated by blank lines.
        # `\A`, not `^`: with MULTILINE, `^` matches after every newline and
        # would collapse this back to line-anchoring.
        r"(?m)(?:\A|\n[ \t]*\n)[ \t]*system[ \t]*:[ \t]*(?=\S)",
        r"<\|im_start\|>"
    )
)


@dataclass(frozen=True, slots=True)
class HeuristicInjectionGuard(InjectionGuardPort):
    def scan(self, text: str) -> InjectionVerdict:
        matches = tuple(m.group(0) for m in (p.search(text) for p in _PATTERNS) if m)
        return InjectionVerdict(risky=bool(matches), matches=matches)
