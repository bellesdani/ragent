import json
import time
import uuid
import asyncio

from collections.abc import AsyncIterator
from app.core.chat import ChatAgentService
from fastapi.responses import StreamingResponse
from app.core.entities import ChatCompletionUsage
from fastapi import APIRouter, Depends, HTTPException, Request
from app.core.entities import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelCard,
    ModelListResponse,
)


router = APIRouter(tags=["Chat"])


def get_chat_service(request: Request) -> ChatAgentService:
    return request.app.state.chat_service


def sse_data(payload: dict[str, object] | str) -> str:
    data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"data: {data}\n\n"


def chunk_text(content: str, size: int = 80) -> list[str]:
    if not content:
        return []
    return [content[index:index + size] for index in range(0, len(content), size)]


async def stream_chat_completion(model: str, content: str, usage: ChatCompletionUsage) -> AsyncIterator[str]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    yield sse_data(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ],
        }
    )

    for chunk in chunk_text(content):
        yield sse_data(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None,
                    }
                ],
            }
        )
        await asyncio.sleep(0.01)

    yield sse_data(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
    )
    yield sse_data(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [],
            "usage": usage.model_dump(),
        }
    )
    yield sse_data("[DONE]")


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models(chat_service: ChatAgentService = Depends(get_chat_service)) -> ModelListResponse:
    return ModelListResponse(
        data=[
            ModelCard(
                id=agent.agent_id,
                name=agent.name,
                description=agent.description,
            )
            for agent in chat_service.list_agents()
        ]
    )


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest, 
    chat_service: ChatAgentService = Depends(get_chat_service),
) -> ChatCompletionResponse | StreamingResponse:
    try:
        result = await chat_service.complete(
            model=request.model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if request.stream:
        # Voy a simular un formato de respuesta en Streaming por temas de compatibilidad con N8N
        return StreamingResponse(
            stream_chat_completion(model=result.model, content=result.content, usage=result.usage),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return ChatCompletionResponse(
        model=result.model,
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionChoiceMessage(content=result.content),
            )
        ],
        usage=result.usage,
    )
