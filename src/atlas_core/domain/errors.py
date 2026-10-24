"""Typed error hierarchy surfaced as a uniform error envelope over HTTP."""

from __future__ import annotations

from typing import Any, ClassVar


class AtlasError(Exception):
    """Base error for all Atlas failures."""

    code: ClassVar[str] = "internal_error"
    status: ClassVar[int] = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def to_payload(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class ConfigError(AtlasError):
    code = "config_error"
    status = 500


class DomainError(AtlasError):
    code = "domain_error"
    status = 400


class NotFoundError(AtlasError):
    code = "not_found"
    status = 404


class InfraError(AtlasError):
    code = "infra_error"
    status = 503


class IngestError(AtlasError):
    code = "ingest_error"
    status = 422


class SecurityError(AtlasError):
    code = "security_error"
    status = 400


class PromptInjectionDetected(SecurityError):
    code = "prompt_injection_detected"
    status = 400


class RateLimitExceeded(AtlasError):
    code = "rate_limited"
    status = 429

    def __init__(self, message: str, *, retry_after: float) -> None:
        super().__init__(message, details={"retry_after_seconds": round(retry_after, 1)})
        self.retry_after = retry_after
