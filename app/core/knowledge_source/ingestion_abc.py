from app.config import Settings
from abc import ABC, abstractmethod
from qdrant_client import AsyncQdrantClient
from app.core.embeddings import EmbeddingClient
from app.core.agent.service import AgentService
from app.core.entities import KnowledgeSourceDefinition


class KnowledgeSourceIngestor(ABC):
    def __init__(self, settings: Settings, agent_service: AgentService, definition: KnowledgeSourceDefinition) -> None:
        self.settings = settings
        self.agent_service = agent_service
        self.embedding_client = EmbeddingClient(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            timeout=settings.llm_timeout_seconds
        )
        self.qdrant_client = AsyncQdrantClient(
            url=settings.qdrant_url, 
            api_key=settings.qdrant_api_key or None
        )
        self.knowledge_source = definition


    @abstractmethod
    async def create_knowledge_source(self):
        raise NotImplementedError


    @abstractmethod
    async def upsert_knowledge_source_data(self, data):
        raise NotImplementedError

