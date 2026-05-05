from typing import Any
from app.config import Settings
from qdrant_client.models import Filter
from qdrant_client import AsyncQdrantClient
from app.core.openai import OpenAICompatClient
from app.core.qdrant_knowledge_sources import QdrantKnowledgeSourceCatalog
from app.core.entities import KnowledgeSource, RetrievalDocument, RetrievedContext


class QdrantRetriever:

    def __init__(self, settings: Settings, embedding_client: OpenAICompatClient) -> None:
        self.default_top_k = 15
        self.settings = settings
        self.embedding_client = embedding_client
        self.client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
        self.knowledge_sources = QdrantKnowledgeSourceCatalog().get_knowledge_sources_by_id()


    async def retrieve(self, query: str, source_ids: list[str], query_filter: Filter | None = None) -> RetrievedContext:
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
            source = self.knowledge_sources[source_id]
            results = await self.client.query_points(
                collection_name=source.collection,
                query=query_vector,
                limit=self.default_top_k,
                with_payload=True,
                query_filter=query_filter,
            )
            points = results.points if hasattr(results, "points") else []
            documents.extend(self._point_to_document(point, source=source) for point in points)

        return RetrievedContext(query=rewritten_query, documents=documents)


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
