from typing import Any
from app.config import Settings
from abc import ABC, abstractmethod
from qdrant_client.models import Filter
from qdrant_client import AsyncQdrantClient
from app.core.embeddings import EmbeddingService
from app.core.knowledge_source.entities import KnowledgeSourceDefinition, RetrievalDocument


class KnowledgeSourceRetrieval(ABC):
    """
    Esta clase define la interfaz base para recuperar documentos de una fuente de conocimiento. Utiliza:
     - Las variables cargadas (Settings)
     - El cliente de embeddings (EmbeddingClient)
     - El cliente de Qdrant (AsyncQdrantClient)

    Funciones publicas:
     - Recuperar documentos relevantes para una consulta (retrieve).
    """

    def __init__(self, settings: Settings, embedding_client: EmbeddingService, qdrant_client: AsyncQdrantClient) -> None:
        self.settings = settings
        self.embedding_client = embedding_client
        self.qdrant_client = qdrant_client


    @abstractmethod
    async def retrieve(
        self,  
        query: str, 
        limit: int, 
        source: KnowledgeSourceDefinition, 
        query_filter: Filter | None = None,
    ) -> list[RetrievalDocument]:
        raise NotImplementedError


    def _point_to_document(self, point: Any, knowledge_source: KnowledgeSourceDefinition) -> RetrievalDocument:
        payload = dict(point.payload)
        if knowledge_source.payload_keys.lexical_content_key:
            content_key = knowledge_source.payload_keys.lexical_content_key
        else:
            content_key = knowledge_source.payload_keys.semantic_content_key 
        return RetrievalDocument(
            id=f"{knowledge_source.id}:{point.id}",
            score=float(point.score),
            content=payload[content_key],
            metadata=payload[knowledge_source.payload_keys.metadata_key],
        )
