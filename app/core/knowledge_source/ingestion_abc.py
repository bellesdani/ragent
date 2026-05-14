from app.config import Settings
from abc import ABC, abstractmethod
from qdrant_client import AsyncQdrantClient
from app.core.utils.embeddings import EmbeddingService
from app.core.knowledge_source.entities import KnowledgeSourceDefinition


class KnowledgeSourceIngestor(ABC):
    """
    Esta clase define la interfaz base para ingestar fuentes de conocimiento. Utiliza:
     - Las variables cargadas (Settings)
     - El servicio de embeddings (EmbeddingService)
     - El cliente de Qdrant (AsyncQdrantClient)
     - La definición de la fuente de conocimiento (KnowledgeSourceDefinition)

    Funciones públicas:
     - Crear la fuente de conocimiento (create_knowledge_source).
     - Añadir datos a la fuente de conocimiento (upsert_knowledge_source_data).
    """

    def __init__(self, settings: Settings, knowledge_source: KnowledgeSourceDefinition) -> None:
        self.settings = settings
        self.embedding_client = EmbeddingService(
            settings=settings,
        )
        self.qdrant_client = AsyncQdrantClient(
            url=settings.qdrant_url, 
            api_key=settings.qdrant_api_key or None
        )
        self.knowledge_source = knowledge_source


    @abstractmethod
    async def create_knowledge_source(self):
        raise NotImplementedError


    @abstractmethod
    async def upsert_knowledge_source_data(self, data):
        raise NotImplementedError
