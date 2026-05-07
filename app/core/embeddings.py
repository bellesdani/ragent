import httpx

from typing import Any

from app.config import Settings


class EmbeddingService:
    """
    Este servicio centraliza la generación de embeddings.
     - Las variables cargadas (Settings)

    Funciones públicas:
     - Generar el embedding de un texto (create_embedding).
    """
    def __init__(self, settings: Settings) -> None:
        headers = {"Content-Type": "application/json"}
        headers["Authorization"] = f"Bearer {settings.embedding_api_key}"
        self.client = httpx.AsyncClient(
            base_url=settings.embedding_base_url.rstrip("/") + "/",
            timeout=settings.llm_timeout_seconds,
            headers=headers,
        )


    async def create_embedding(self, input_text: str, model: str) -> list[float]:
        payload = {"model": model, "input": input_text}
        response = await self.client.post("embeddings", json=payload)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data["data"][0]["embedding"]
