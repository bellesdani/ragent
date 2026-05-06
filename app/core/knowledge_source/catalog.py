from app.core.entities import KnowledgeSourceDefinition
from app.core.knowledge_source.ingestion_tickets import TicketsKnowledgeSourceIngestion


class KnowledgeSourceCatalog():
    def __init__(self) -> None:
        self._knowledge_sources = {
            "devices": KnowledgeSourceDefinition(
                id="devices",
                name="Devices",
                description="Información sobre dispositivos de la empresa, como servidores y equipos de usuario y planta.",
                collection="devices",
                vector_name=None,
                ingestion_module=None,
                retrieval_module=None,
            ),
            "employees": KnowledgeSourceDefinition(
                id="employees",
                name="Employees",
                description="Información sobre empleados y su contacto corporativo: correo, teléfono y extensión.",
                collection="employees",
                vector_name=None,
                ingestion_module=None,
                retrieval_module=None,
            ),
            "manuals": KnowledgeSourceDefinition(
                id="manuals",
                name="Manuals",
                description="Información sobre manuales de software y operativas habituales dentro de la empresa.",
                collection="manuals",
                vector_name=None,
                ingestion_module=None,
                retrieval_module=None,
            ),
            "tickets": KnowledgeSourceDefinition(
                id="tickets",
                name="Tickets",
                description="Información sobre los tickets registrados en Helpdesk.",
                collection="tickets",
                vector_name="dense_vector",
                ingestion_module=TicketsKnowledgeSourceIngestion(),
                retrieval_module=None,
            ),
        }


    def list_knowledge_sources(self) -> list[KnowledgeSourceDefinition]:
        return [
            knowledge_source for knowledge_source in self._knowledge_sources.values()
        ]


    def get_knowledge_source(self, knowledge_source_id: str) -> KnowledgeSourceDefinition:
        if knowledge_source_id in self._knowledge_sources:
            return self._knowledge_sources[knowledge_source_id]
        raise ValueError(f"Fuente de conocimiento '{knowledge_source_id}' desconocida. Agentes disponibles: {', '.join(sorted(self._knowledge_sources))}")
    
