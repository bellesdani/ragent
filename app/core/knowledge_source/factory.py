from app.config import Settings
from app.core.agent.service import AgentService
from app.core.entities import KnowledgeSourceDefinition
from app.core.knowledge_source.ingestion_abc import KnowledgeSourceIngestion
from app.core.knowledge_source.retrieval import KnowledgeSourceRetriever


class KnowledgeSourceFactory:
    def __init__(self, settings: Settings, agent_service: AgentService) -> None:
        self.settings = settings
        self.agent_service = agent_service


    def build(self, definition: KnowledgeSourceDefinition) -> tuple[KnowledgeSourceIngestion, KnowledgeSourceRetriever]:
        return definition.ingestion_module, definition.retrieval_module