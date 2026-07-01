from qdrant_client.models import Filter
from app.core.knowledge_source.retrieval.abc import KnowledgeSourceRetrieval
from app.core.knowledge_source.entities import KnowledgeSourceDefinition, RetrievalDocument


class SemanticKnowledgeSourceRetrieval(KnowledgeSourceRetrieval):
    """
    Esta clase recupera documentos mediante búsqueda semántica.
     - Los vectores densos son las representaciones semánticas.
    """

    async def retrieve(
            self, 
            query: str, 
            limit: int,
            source: KnowledgeSourceDefinition, 
            query_filter: Filter | None = None
    ) -> list[RetrievalDocument]:
        # Buscamos los puntos con mayor similitud semantica a partir de los embeddings del texto semántico
        results = await self.qdrant_client.query_points(
            collection_name=source.collection_name,
            query=await self.embedding_client.create_embedding(
                input_text=query,
                model=self.settings.embedding_model,
            ),
            using=source.dense_vector_name,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        )
        # Convertimos los puntos en "Documentos" para estandarizar la salida
        points = results.points if hasattr(results, "points") else []
        return [
            self._point_to_document(point, knowledge_source=source) 
            for point in points
        ]
