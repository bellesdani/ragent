from __future__ import annotations

from typing import Any
from app.core.config import Settings
from qdrant_client import AsyncQdrantClient
from app.core.openai import OpenAICompatClient
from app.core.entities import KnowledgeSource, RetrievalDocument, RetrievedContext


DEFAULT_TOP_K = 15
DEFAULT_SEARCH_SOURCES = (
    KnowledgeSource(
        id="devices",
        name="Devices",
        description="Informacion sobre dispositivos de la empresa, como servidores y equipos de usuario y planta.",
        collection="devices",
    ),
    KnowledgeSource(
        id="employees",
        name="Employees",
        description="Informacion sobre empleados y su contacto corporativo: correo, telefono y extension.",
        collection="employees",
    ),
    KnowledgeSource(
        id="manuals",
        name="Manuals",
        description="Informacion sobre manuales de software y operativas habituales dentro de la empresa.",
        collection="manuals",
    ),
)




class QdrantRetriever:

    def __init__(self, settings: Settings, embedding_client: OpenAICompatClient) -> None:
        self.settings = settings
        self.embedding_client = embedding_client
        self.client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
        self.sources = {source.id: source for source in DEFAULT_SEARCH_SOURCES}


    async def retrieve(self, query: str, source_ids: list[str]) -> RetrievedContext:
        # Limpieza de la query básica
        rewritten_query = self._rewrite_query(query)

        # Creamos el embedding
        query_vector = await self.embedding_client.create_embedding(
            input_text=rewritten_query,
            model=self.settings.embedding_model,
        )

        # En cada fuente/colección de source_ids, buscamos los puntos con mayor similitud semántica
        #  y generamos para cada punto obtenido, un documento: un punto referenciable con su payload y metadata
        documents = []
        for source_id in source_ids:
            source = self.sources[source_id]
            results = await self.client.query_points(
                collection_name=source.collection,
                query=query_vector,
                limit=DEFAULT_TOP_K,
                with_payload=True,
            )
            points = results.points if hasattr(results, "points") else []
            documents.extend(self._point_to_document(point, source=source) for point in points)

        return RetrievedContext(query=rewritten_query, documents=documents)


    def list_sources(self) -> list[dict[str, str]]:
        return [
            {
                "id": source.id,
                "name": source.name,
                "description": source.description,
            }
            for source in self.sources.values()
        ]


    def _rewrite_query(self, query: str) -> str:
        return query.strip()


    def _point_to_document(self, point: Any, source: KnowledgeSource) -> RetrievalDocument:
        payload = dict(point.payload)
        return RetrievalDocument(
            id=f"{source.id}:{point.id}",
            score=float(point.score),
            text=payload["content"],
            metadata=payload["metadata"],
        )