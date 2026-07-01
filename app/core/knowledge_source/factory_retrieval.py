from app.config import Settings
from qdrant_client import AsyncQdrantClient
from app.core.utils.embeddings import EmbeddingService
from app.core.knowledge_source.entities import KnowledgeSourceDefinition
from app.core.knowledge_source.retrieval_abc import KnowledgeSourceRetrieval
from app.core.knowledge_source.retrieval_hybrid import HybridKnowledgeSourceRetrieval
from app.core.knowledge_source.retrieval_lexical import LexicalKnowledgeSourceRetrieval
from app.core.knowledge_source.retrieval_semantic import SemanticKnowledgeSourceRetrieval


class KnowledgeSourceRetrievalFactory:
    """
    Esta factoría selecciona la estrategia de búsqueda configurada para cada fuente de conocimiento.

    Funciones públicas:
     - Construir el servicio de búsqueda correspondiente (build).
    """

    def __init__(self, settings: Settings, embedding_client: EmbeddingService, qdrant_client: AsyncQdrantClient) -> None:
        self.retrievers: dict[str, KnowledgeSourceRetrieval] = {
            "semantic": SemanticKnowledgeSourceRetrieval(
                settings=settings,
                embedding_client=embedding_client,
                qdrant_client=qdrant_client,
            ),
            "hybrid": HybridKnowledgeSourceRetrieval(
                settings=settings,
                embedding_client=embedding_client,
                qdrant_client=qdrant_client,
            ),
            "lexical": LexicalKnowledgeSourceRetrieval(
                settings=settings,
                embedding_client=embedding_client,
                qdrant_client=qdrant_client,
            ),
        }


    def build(self, definition: KnowledgeSourceDefinition) -> KnowledgeSourceRetrieval:
        retriever = self.retrievers.get(definition.retrieval_type)
        if retriever is None:
            raise ValueError(
                f"La fuente de conocimiento '{definition.id}' tiene un tipo de búsqueda no soportado: {definition.retrieval_type}"
            )
        return retriever
