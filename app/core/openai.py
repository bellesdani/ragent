from __future__ import annotations

import httpx
import logging

from typing import Any


logger = logging.getLogger(__name__)


class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str, timeout: float) -> None:
        headers = {"Content-Type": "application/json"}
        headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout,
            headers=headers,
        )

    async def create_embedding(self, input_text: str, model: str) -> list[float]:
        payload = {"model": model, "input": input_text}
        response = await self.client.post("embeddings", json=payload)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data["data"][0]["embedding"]
