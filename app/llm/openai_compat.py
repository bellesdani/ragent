from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.api.schemas.openai import ChatCompletionUsage, LLMChatResult, LLMToolCall, LLMToolFunction, ModelCard


logger = logging.getLogger(__name__)


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
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMChatResult:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        logger.debug(
            "Sending chat completion request | provider=%s model=%s messages=%d tools=%s tool_choice=%s",
            self.provider,
            model,
            len(messages),
            bool(tools),
            tool_choice,
        )
        response = await self.client.post("chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]
        normalized_content = self._normalize_content(message.get("content"))
        logger.debug(
            "Received chat completion response | provider=%s model=%s has_content=%s tool_calls=%d",
            self.provider,
            model,
            bool(normalized_content),
            len(message.get("tool_calls", [])),
        )
        if not normalized_content and not message.get("tool_calls"):
            logger.warning(
                "Provider returned chat completion without content | provider=%s model=%s raw_message=%s",
                self.provider,
                model,
                json.dumps(message, ensure_ascii=False, default=str),
            )
        usage = data.get("usage") or {}
        return LLMChatResult(
            content=normalized_content,
            usage=ChatCompletionUsage(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                total_tokens=int(usage.get("total_tokens", 0)),
            ),
            tool_calls=[
                LLMToolCall(
                    id=item["id"],
                    type=item.get("type", "function"),
                    function=LLMToolFunction(
                        name=item["function"]["name"],
                        arguments=item["function"]["arguments"],
                    ),
                )
                for item in message.get("tool_calls", [])
            ],
        )

    async def stream_chat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        logger.debug(
            "Sending streaming chat completion request | provider=%s model=%s messages=%d",
            self.provider,
            model,
            len(messages),
        )
        async with self.client.stream("POST", "chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid streaming chunk | raw=%s", data)
                    continue
                content = self._extract_stream_content(payload)
                if content:
                    yield content

    def _normalize_content(self, content: Any) -> str | None:
        if content is None:
            return None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            fragments: list[str] = []
            for item in content:
                if isinstance(item, str):
                    fragments.append(item)
                    continue
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        fragments.append(text_value)
                    elif isinstance(text_value, dict) and isinstance(text_value.get("value"), str):
                        fragments.append(text_value["value"])
                elif isinstance(item.get("content"), str):
                    fragments.append(item["content"])
                elif isinstance(item.get("value"), str):
                    fragments.append(item["value"])
            merged = "".join(fragments).strip()
            return merged or None
        if isinstance(content, dict):
            for key in ("text", "content", "value"):
                value = content.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        return str(content)

    def _extract_stream_content(self, payload: dict[str, Any]) -> str | None:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        choice = choices[0] or {}
        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = self._normalize_content(delta.get("content"))
            if content:
                return content
        message = choice.get("message")
        if isinstance(message, dict):
            content = self._normalize_content(message.get("content"))
            if content:
                return content
        text = choice.get("text")
        if isinstance(text, str) and text:
            return text
        return None

    async def create_embedding(self, input_text: str, model: str) -> list[float]:
        payload = {"model": model, "input": input_text}
        logger.debug(
            "Sending embedding request | provider=%s model=%s input=%s",
            self.provider,
            model,
            input_text[:200],
        )
        response = await self.client.post("embeddings", json=payload)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        logger.debug(
            "Received embedding response | provider=%s model=%s dimensions=%d",
            self.provider,
            model,
            len(data["data"][0]["embedding"]),
        )
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
