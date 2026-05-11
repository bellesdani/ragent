from qdrant_client import models
from qdrant_client.models import Filter
from app.core.knowledge_source.retrieval_abc import KnowledgeSourceRetrieval
from app.core.knowledge_source.entities import KnowledgeSourceDefinition, RetrievalDocument


class HybridKnowledgeSourceRetrieval(KnowledgeSourceRetrieval):
    """
    Esta clase recupera documentos mediante busqueda híbrida.
     - Los vectores densos son los embeddings de representaciones semánticas.
     - Los vectores dispersos son las representaciones léxicas.
    """

    async def retrieve(
            self, 
            query: str, 
            limit: int,
            source: KnowledgeSourceDefinition, 
            query_filter: Filter | None = None,
        ) -> list[RetrievalDocument]:
        # Buscamos por similitud semantica y por coincidencia lexica, y fusionamos los resultados
        #  - La búsqueda semántica no cambia respecto a otros casos: obtenemos los embeddings más próximos a la query
        #  - Para la búsqueda léxica, utilzamos bm25. Es el estándar actual
        results = await self.qdrant_client.query_points(
            collection_name=source.collection_name,
            prefetch=[
                models.Prefetch(
                    query=await self.embedding_client.create_embedding(
                        input_text=query,
                        model=self.settings.embedding_model,
                    ),
                    using=source.dense_vector_name,
                    filter=query_filter,
                    limit=limit,
                ),
                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model="Qdrant/bm25",
                    ),
                    using=source.sparse_vector_name,
                    filter=query_filter,
                    limit=limit,
                ),
            ],
            # Para fusionar los resultados vamos a usar Reciprocal Rank Fusion (RRF)
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        # Convertimos los puntos en "Documentos" para estandarizar la salida
        points = results.points if hasattr(results, "points") else []
        return [
            self._point_to_document(point, knowledge_source=source) 
            for point in points
        ]
