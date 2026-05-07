from app.config import Settings
from qdrant_client.models import Filter
from qdrant_client import AsyncQdrantClient
from app.core.embeddings import EmbeddingService
from app.core.knowledge_source.entities import RetrievedContext
from app.core.knowledge_source.catalog import KnowledgeSourceCatalog
from app.core.knowledge_source.retrieval_abc import KnowledgeSourceRetrieval
from app.core.knowledge_source.retrieval_hybrid import HybridKnowledgeSourceRetrieval
from app.core.knowledge_source.retrieval_semantic import SemanticKnowledgeSourceRetrieval


class KnowledgeSourceRetrievalService:
    """
    Este servicio busca contexto en las fuentes de conocimiento. Utiliza:
     - Las variables cargadas (Settings)
     - El cliente de embeddings (EmbeddingClient)
     - El cliente de Qdrant (AsyncQdrantClient)
     - El catálogo de fuentes de conocimiento (KnowledgeSourceCatalog)

    Funciones públicas:
     - Recuperar documentos relevantes para una consulta (retrieve).
    """

    def __init__(self, settings: Settings, embedding_client: EmbeddingService) -> None:
        self.settings = settings
        self.embedding_client = embedding_client
        self.qdrant_client = AsyncQdrantClient(
            url=settings.qdrant_url, 
            api_key=settings.qdrant_api_key or None
        )
        self.knowledge_source_catalog = KnowledgeSourceCatalog()
        self.retrievers: dict[str, KnowledgeSourceRetrieval] = {
            "semantic": SemanticKnowledgeSourceRetrieval(
                settings=settings,
                embedding_client=embedding_client,
                qdrant_client=self.qdrant_client,
            ),
            "hybrid": HybridKnowledgeSourceRetrieval(
                settings=settings,
                embedding_client=embedding_client,
                qdrant_client=self.qdrant_client,
            ),
        }


    async def retrieve(
            self, 
            query: str, 
            limit: int,
            source_ids: list[str], 
            query_filter: Filter | None = None
    ) -> RetrievedContext:
        # Limpieza de la query básica
        rewritten_query = self._rewrite_query(query)

        # En cada fuente de source_ids, usamos la estrategia configurada
        #  y generamos para cada punto obtenido, un documento: un punto referenciable con su payload y metadata
        documents = []
        for source_id in source_ids:
            source = self.knowledge_source_catalog.get_knowledge_source(source_id)
            retriever = self.retrievers.get(source.retrieval_type)
            if retriever is None:
                raise ValueError(f"La fuente de conocimiento '{source.id}' tiene un tipo de búsqueda no soportado: {source.retrieval_type}")
            documents.extend(
                await retriever.retrieve(
                    query=rewritten_query,
                    source=source,
                    limit=limit,
                    query_filter=query_filter,
                )
            )

        return RetrievedContext(query=rewritten_query, documents=documents)


    def _rewrite_query(self, query: str) -> str:
        return query.strip()
