"""HTTP surface: liveness and auth gating (no external services needed)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from atlas_api.main import create_app
from atlas_core.config import AtlasSettings


def _client() -> TestClient:
    settings = AtlasSettings(security={"api_keys": "test-key=tenant-a"})
    return TestClient(create_app(settings))


def test_healthz() -> None:
    with _client() as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Request-ID"]


def test_chat_requires_api_key() -> None:
    with _client() as client:
        response = client.post("/v1/chat", json={"query": "hello"})
    assert response.status_code == 401


def test_chat_rejects_unknown_api_key() -> None:
    with _client() as client:
        response = client.post("/v1/chat", json={"query": "hello"}, headers={"X-API-Key": "nope"})
    assert response.status_code == 401
