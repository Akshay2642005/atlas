"""Chat endpoint: grounded answer (sync) or SSE stream."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Security
from fastapi.responses import JSONResponse, StreamingResponse

from atlas_api.deps import AnswerServiceDep, Authenticated
from atlas_api.schemas import AnswerOut, ChatIn
from atlas_api.security import API_KEY_SCHEME, AUTH_RESPONSES, BEARER_SCHEME
from atlas_core.domain.models import Query

router = APIRouter(
    prefix="/v1",
    tags=["chat"],
    dependencies=[Security(API_KEY_SCHEME), Security(BEARER_SCHEME)],
    responses=AUTH_RESPONSES,
)


@router.post("/chat", summary="Ask a grounded question over tenant documents", response_model=None)
async def chat(
    body: ChatIn,
    tenant: Authenticated,
    service: AnswerServiceDep,
) -> JSONResponse | StreamingResponse:
    query = Query(id=uuid.uuid4(), tenant_id=tenant.tenant_id, text=body.query)
    if body.stream:
        return StreamingResponse(_sse(service.stream(query)), media_type="text/event-stream")
    answer = await service.answer(query)
    return JSONResponse(AnswerOut.from_domain(answer).model_dump(mode="json"))


async def _sse(events: AsyncIterator[dict[str, object]]) -> AsyncIterator[str]:
    async for event in events:
        yield f"data: {json.dumps(event)}\n\n"
    yield "data: [DONE]\n\n"
