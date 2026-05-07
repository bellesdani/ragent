from qdrant_client.models import Filter
from app.core.entities import KnowledgeSourceDefinition, RetrievalDocument
from app.core.knowledge_source.retrieval_abc import KnowledgeSourceRetrieval


class SemanticKnowledgeSourceRetrieval(KnowledgeSourceRetrieval):
    """
    Esta clase recupera documentos mediante busqueda semántica.
    """

    async def retrieve(
            self, 
            query: str, 
            limit: int,
            source: KnowledgeSourceDefinition, 
            query_filter: Filter | None = None
    ) -> list[RetrievalDocument]:
        # Creamos el embedding
        query_vector = await self.embedding_client.create_embedding(
            input_text=query,
            model=self.settings.embedding_model,
        )

        # Buscamos los puntos con mayor similitud semantica
        results = await self.qdrant_client.query_points(
            collection_name=source.collection_name,
            query=query_vector,
            using=source.dense_vector_name,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        )
        points = results.points if hasattr(results, "points") else []
        return [self._point_to_document(point, source=source) for point in points]
