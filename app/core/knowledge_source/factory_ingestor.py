from app.config import Settings
from app.core.agent.service import AgentService
from app.core.knowledge_source.entities import KnowledgeSourceDefinition
from app.core.knowledge_source.ingestion_abc import KnowledgeSourceIngestor
from app.core.knowledge_source.ingestion_devices import DevicesKnowledgeSourceIngestor
from app.core.knowledge_source.ingestion_tickets import TicketsKnowledgeSourceIngestor
from app.core.knowledge_source.ingestion_articles import ArticlesKnowledgeSourceIngestor
from app.core.knowledge_source.ingestion_employees import EmployeesKnowledgeSourceIngestor
from app.core.knowledge_source.ingestion_html_manuals import HtmlManualsKnowledgeSourceIngestor


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
                knowledge_source=definition,
            )
        elif definition.id == "devices":
            return DevicesKnowledgeSourceIngestor(
                settings=self.settings,
                knowledge_source=definition,
            )
        elif definition.id == "employees":
            return EmployeesKnowledgeSourceIngestor(
                settings=self.settings,
                knowledge_source=definition,
            )
        elif definition.id == "manuals":
            return HtmlManualsKnowledgeSourceIngestor(
                settings=self.settings,
                knowledge_source=definition,
            )
        elif definition.id == "articles":
            return ArticlesKnowledgeSourceIngestor(
                settings=self.settings,
                knowledge_source=definition,
            )
        raise ValueError(f"{definition.id} no tiene un modelo de ingesta definido")
