from __future__ import annotations

import httpx

from typing import Any
from app.api.schemas.openai import ChatCompletionUsage, LLMChatResult, ModelCard


class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str, timeout: float, provider: str) -> None:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.provider = provider
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout,
            headers=headers,
        )

    async def create_chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> LLMChatResult:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        response = await self.client.post("chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        return LLMChatResult(
            content=choice,
            usage=ChatCompletionUsage(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                total_tokens=int(usage.get("total_tokens", 0)),
            ),
        )

    async def create_embedding(self, input_text: str, model: str) -> list[float]:
        payload = {"model": model, "input": input_text}
        response = await self.client.post("embeddings", json=payload)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data["data"][0]["embedding"]

    async def list_models(self) -> list[ModelCard]:
        response = await self.client.get("models")
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        models = []
        for item in data.get("data", []):
            model_id = item.get("id")
            if not model_id:
                continue
            models.append(
                ModelCard(
                    id=model_id,
                    created=int(item.get("created") or 0),
                    owned_by=item.get("owned_by") or self.provider,
                )
            )
        return models
