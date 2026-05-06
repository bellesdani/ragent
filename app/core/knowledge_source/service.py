from app.config import Settings
from app.core.agent.service import AgentService
from app.core.entities import KnowledgeSource
from app.core.knowledge_source.catalog import KnowledgeSourceCatalog
from app.core.knowledge_source.ingestion import QdrantIngestor


class KnowledgeSourceService():
    """
    Este servicio es el punto de acceso y gestor de fuentes de conocimiento. Utiliza:
     - Las variables cargadas (Settings)
     - El catálogo de fuentes de conocimiento (KnowledgeSourceCatalog)
     - El servicio de ingesta de Qdrant (QdrantIngestor)
    
    Funciones públicas:
     - Listar las fuentes de conocimiento disponibles (list_knowledge_sources).
     - Crear una nueva fuente de conocimiento (create_knowledge_source).
     - Añadir datos a una fuente de conocimiento (upsert_knowledge_source_data).
    """
    
    def __init__(self, settings: Settings, agent_service: AgentService) -> None:
        self.settings = settings
        self.knowledge_source_catalog = KnowledgeSourceCatalog()
        self.qdrant_ingestor = QdrantIngestor(
            settings=settings,
            agent_service=agent_service
        )


    def list_knowledge_sources(self) -> list[KnowledgeSource]:
        return self.knowledge_source_catalog.list_knowledge_sources()
    

    async def create_knowledge_source(self, knowledge_source_id: str):
        if knowledge_source_id == 'tickets':
            return self.qdrant_ingestor.create_tickets_collection()
        else:
            raise ValueError(f"No se disponde de la fuente de conocimiento '{knowledge_source_id}' en el catálogo.")


    async def upsert_knowledge_source_data(self, knowledge_source_id: str, data):
        if knowledge_source_id == 'tickets':
            return self.qdrant_ingestor.upsert_tickets_points(data)
        else:
            raise ValueError(f"No se disponde de la fuente de conocimiento '{knowledge_source_id}' en el catálogo.")