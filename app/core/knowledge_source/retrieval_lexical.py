from qdrant_client import models
from qdrant_client.models import Filter
from app.core.knowledge_source.retrieval_abc import KnowledgeSourceRetrieval
from app.core.knowledge_source.entities import KnowledgeSourceDefinition, RetrievalDocument


class HybridKnowledgeSourceRetrieval(KnowledgeSourceRetrieval):
    """
    Esta clase recupera documentos mediante búsqueda léxica.
     - Los vectores dispersos son las representaciones léxicas.
    """

    async def retrieve(
            self, 
            query: str, 
            limit: int,
            source: KnowledgeSourceDefinition, 
            query_filter: Filter | None = None,
        ) -> list[RetrievalDocument]:
        #  Buscamos los puntos con mayor similitud léxica a partir del texto léxico
        results = await self.qdrant_client.query_points(
            collection_name=source.collection_name,
            query=models.Document(
                text=query,
                model="Qdrant/bm25",
            ),
            using=source.sparse_vector_name,
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
