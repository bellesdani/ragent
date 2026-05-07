from app.config import Settings
from app.core.agent.service import AgentService
from app.core.knowledge_source.entities import KnowledgeSourceDefinition
from app.core.knowledge_source.ingestion_abc import KnowledgeSourceIngestor
from app.core.knowledge_source.ingestion_tickets import TicketsKnowledgeSourceIngestor


class KnowledgeSourceIngestorFactory:
    """
    Esta factoría construye servicios de ingesta para cada fuente de conocimiento. Utiliza:
     - Las variables cargadas (Settings)
     - El servicio de agentes (AgentService), ya que algún proceso de ingesta puede requerir la ayuda de agentes.

    Funciones públicas:
     - Construir el servicio de ingesta correspondiente (build).
    """

    def __init__(self, settings: Settings, agent_service: AgentService) -> None:
        self.settings = settings
        self.agent_service = agent_service


    def build(self, definition: KnowledgeSourceDefinition) -> KnowledgeSourceIngestor:
        if definition.id == "tickets":
            return TicketsKnowledgeSourceIngestor(
                settings=self.settings,
                agent_service=self.agent_service,
                definition=definition,
            )
        raise ValueError(f"{definition.id} no tiene un modelo de ingesta definido")
