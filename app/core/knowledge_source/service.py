from app.config import Settings
from app.core.agent.service import AgentService
from app.core.knowledge_source.catalog import KnowledgeSourceCatalog
from app.core.knowledge_source.entities import KnowledgeSourceDefinition
from app.core.knowledge_source.factory_ingestor import KnowledgeSourceIngestorFactory


class KnowledgeSourceService():
    """
    Este servicio es el punto de acceso y gestor de fuentes de conocimiento. Utiliza:
     - Las variables cargadas (Settings)
     - El catálogo de fuentes de conocimiento (KnowledgeSourceCatalog)
     - El factory de servicios de ingesta de Qdrant (KnowledgeSourceIngestorFactory)
    
    Funciones públicas:
     - Listar las fuentes de conocimiento disponibles (list_knowledge_sources).
     - Crear una nueva fuente de conocimiento (create_knowledge_source).
     - Añadir datos a una fuente de conocimiento (upsert_knowledge_source_data).
    """
    
    def __init__(self, settings: Settings, agent_service: AgentService) -> None:
        self.settings = settings
        self.catalog = KnowledgeSourceCatalog()
        self.ingestor_factory = KnowledgeSourceIngestorFactory(
            settings=settings,
            agent_service=agent_service
        )


    def list_knowledge_sources(self) -> list[KnowledgeSourceDefinition]:
        return self.catalog.list_knowledge_sources()
    

    async def create_knowledge_source(self, knowledge_source_id: str):
        definition = self.catalog.get_knowledge_source(knowledge_source_id)
        ingestor = self.ingestor_factory.build(definition)
        return await ingestor.create_knowledge_source()


    async def upsert_knowledge_source_data(self, knowledge_source_id: str, data):
        definition = self.catalog.get_knowledge_source(knowledge_source_id)
        ingestor = self.ingestor_factory.build(definition)
        return await ingestor.upsert_knowledge_source_data(data)
