from qdrant_client import models
from qdrant_client.models import Filter
from app.core.entities import KnowledgeSourceDefinition, RetrievalDocument
from app.core.knowledge_source.retrieval_abc import KnowledgeSourceRetrieval


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
        if source.dense_vector_name is None:
            raise ValueError(f"La fuente de conocimiento '{source.id}' no tiene vector denso configurado.")
        if source.sparse_vector_name is None:
            raise ValueError(f"La fuente de conocimiento '{source.id}' no tiene vector disperso configurado.")

        # Creamos el embedding
        query_vector = await self.embedding_client.create_embedding(
            input_text=query,
            model=self.settings.embedding_model,
        )

        # Buscamos por similitud semantica y por coincidencia lexica, y fusionamos los resultados
        results = await self.qdrant_client.query_points(
            collection_name=source.collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_vector,
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
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        points = results.points if hasattr(results, "points") else []
        return [self._point_to_document(point, source=source) for point in points]
