import json
import time
import uuid
import asyncio

from collections.abc import AsyncIterator
from app.core.chat.entities import ChatCompletionUsage


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
