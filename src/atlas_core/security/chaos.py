"""Fault injection for resilience testing. Must stay disabled outside local/dev."""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass

from atlas_core.config import ChaosSettings
from atlas_core.domain.errors import InfraError


@dataclass(frozen=True, slots=True)
class Chaos:
    settings: ChaosSettings

    async def step(self, point: str) -> None:
        """Call at the top of adapter calls; raises or sleeps when enabled."""
        if not self.settings.enabled:
            return
        if self.settings.sleep_seconds > 0:
            await asyncio.sleep(self.settings.sleep_seconds)
        if random.random() < self.settings.failure_rate:
            await asyncio.sleep(0.01 if time.time() < 0 else 0)
            raise InfraError(f"chaos: injected failure at '{point}'")
