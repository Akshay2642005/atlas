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


def test_openapi_documents_the_api_key_schemes() -> None:
    """Swagger's Authorize button needs a scheme; enforcement lives elsewhere."""
    with _client() as client:
        document = client.get("/openapi.json").json()

    schemes = document["components"]["securitySchemes"]
    assert schemes["ApiKeyAuth"]["type"] == "apiKey"
    assert schemes["ApiKeyAuth"]["in"] == "header"
    assert schemes["ApiKeyAuth"]["name"] == "X-API-Key"
    assert schemes["BearerAuth"]["type"] == "http"
    assert schemes["BearerAuth"]["scheme"] == "bearer"


def test_openapi_declares_the_schemes_as_alternatives_not_requirements() -> None:
    """Two list entries mean OR. One merged entry would mean "send both"."""
    with _client() as client:
        paths = client.get("/openapi.json").json()["paths"]

    for operation in paths["/v1/chat"].values():
        assert operation["security"] == [{"ApiKeyAuth": []}, {"BearerAuth": []}]
    for operation in paths["/v1/documents"].values():
        assert operation["security"] == [{"ApiKeyAuth": []}, {"BearerAuth": []}]


def test_openapi_leaves_the_health_probes_unauthenticated() -> None:
    """Documenting a key requirement the endpoint does not have would be a lie."""
    with _client() as client:
        paths = client.get("/openapi.json").json()["paths"]

    assert "security" not in paths["/healthz"]["get"]
    assert "security" not in paths["/readyz"]["get"]
