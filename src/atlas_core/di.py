"""Dishka providers shared by every entrypoint."""

from __future__ import annotations

from dishka import Provider, Scope, provide

from atlas_core.config import AtlasSettings, load_settings
from atlas_core.security.chaos import Chaos
from atlas_core.security.injection import HeuristicInjectionGuard
from atlas_core.security.pii import RegexPiiDetector
from atlas_core.security.sanitize import Sanitizer


class CoreProvider(Provider):
    def __init__(self, settings_override: AtlasSettings | None = None) -> None:
        super().__init__()
        self._settings_override = settings_override

    @provide(scope=Scope.APP)
    def settings(self) -> AtlasSettings:
        return self._settings_override or load_settings()

    @provide(scope=Scope.APP)
    def chaos(self, settings: AtlasSettings) -> Chaos:
        return Chaos(settings.chaos)

    @provide(scope=Scope.APP)
    def pii_detector(self) -> RegexPiiDetector:
        return RegexPiiDetector()

    @provide(scope=Scope.APP)
    def injection_guard(self, settings: AtlasSettings) -> HeuristicInjectionGuard:
        return HeuristicInjectionGuard()

    @provide(scope=Scope.APP)
    def sanitizer(self) -> Sanitizer:
        return Sanitizer()


def providers(settings_override: AtlasSettings | None = None) -> list[Provider]:
    return [CoreProvider(settings_override)]
