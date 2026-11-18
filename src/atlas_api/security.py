"""OpenAPI description of the API-key schemes.

Nothing here authenticates. `atlas_api.middlewares.authenticated` is the single
enforcement point, and it accepts an `X-API-Key` header or an
`Authorization: Bearer` token. What this module adds is the *documentation* that
lets Swagger's Authorize button attach either credential to the "Try it out"
requests, which is otherwise impossible to discover from `/docs`.

The schemes are declared with `auto_error=False` and attached to a router as
`Security(...)` dependencies only — `Security` is what makes FastAPI emit them
into `components.securitySchemes`, and `auto_error=False` is what stops that
declaration from becoming a second, weaker gate that disagrees with the
middleware. Declaring a scheme is not enforcing it.
"""

from __future__ import annotations

from typing import Any

from fastapi.security import APIKeyHeader, HTTPBearer

from atlas_api.schemas import ErrorEnvelope

API_KEY_HEADER = "X-API-Key"

API_KEY_SCHEME = APIKeyHeader(
    name=API_KEY_HEADER,
    scheme_name="ApiKeyAuth",
    description=(
        "Tenant API key, sent verbatim in the `X-API-Key` header. "
        'Keys map to tenants through SECURITY__API_KEYS="key=tenant".'
    ),
    auto_error=False,
)

BEARER_SCHEME = HTTPBearer(
    scheme_name="BearerAuth",
    description=(
        "The same tenant API key, sent as `Authorization: Bearer <key>`. "
        "Not a JWT: Atlas does not issue tokens, it maps static keys to tenants."
    ),
    auto_error=False,
)

# Two entries, not one object: OpenAPI reads a list as OR. Merging them into a
# single requirement would document a client as needing both credentials.
AUTH_SCHEMES = [API_KEY_SCHEME, BEARER_SCHEME]

UNAUTHORIZED_RESPONSE: dict[str, Any] = {
    "model": ErrorEnvelope,
    "description": "Missing or unknown API key.",
    "content": {
        "application/json": {
            "example": {
                "error": {
                    "code": "unauthorized",
                    "message": "missing API key",
                    "details": None,
                }
            },
        }
    },
}

# Applied per router rather than app-wide, because `/healthz` and `/readyz` are
# deliberately unauthenticated and documenting them otherwise would be a lie.
AUTH_RESPONSES: dict[int | str, dict[str, Any]] = {401: UNAUTHORIZED_RESPONSE}
