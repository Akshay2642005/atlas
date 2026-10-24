"""Structured logging + OpenTelemetry tracing setup."""

from __future__ import annotations

import logging
import sys

import structlog
from opentelemetry import trace
from opentelemetry.trace import Tracer
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from atlas_core.config import ObservabilitySettings

_configured = False


def setup_logging(settings: ObservabilitySettings) -> None:
    global _configured
    if _configured:
        return
    _configured = True
    logging.basicConfig(level=settings.log_level, stream=sys.stdout, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer() if not settings.log_json else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(settings.log_level)),
        cache_logger_on_first_use=True,
    )


def setup_tracing(settings: ObservabilitySettings) -> trace.TracerProvider | None:
    if not settings.otlp_endpoint:
        return None
    provider = TracerProvider(resource=Resource(attributes={"service.name": settings.service_name}))
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter  # noqa: PLC0415

        exporter: BatchSpanProcessor = BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otlp_endpoint))
    except ImportError:
        exporter = BatchSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(exporter)
    trace.set_tracer_provider(provider)
    return provider


def get_tracer(name: str) -> Tracer:
    return trace.get_tracer(name)
