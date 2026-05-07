from app.core.entities import KnowledgeSourceDefinition


class KnowledgeSourceCatalog():
    """
    Este catálogo centraliza las fuentes de conocimiento disponibles. Utiliza:
     - Las definiciones de fuentes de conocimiento (KnowledgeSourceDefinition)

    Funciones públicas:
     - Listar las fuentes de conocimiento disponibles (list_knowledge_sources).
     - Obtener una fuente de conocimiento por identificador (get_knowledge_source).
    """

    def __init__(self) -> None:
        self._knowledge_sources = {
            "devices": KnowledgeSourceDefinition(
                id="devices",
                name="Devices",
                description="Información sobre dispositivos de la empresa, como servidores y equipos de usuario y planta.",
                retrieval_type="semantic",
                collection_name="devices",
                dense_vector_name=None,
                sparse_vector_name=None,
            ),
            "employees": KnowledgeSourceDefinition(
                id="employees",
                name="Employees",
                description="Información sobre empleados y su contacto corporativo: correo, teléfono y extensión.",
                retrieval_type="semantic",
                collection_name="employees",
                dense_vector_name=None,
                sparse_vector_name=None,
            ),
            "manuals": KnowledgeSourceDefinition(
                id="manuals",
                name="Manuals",
                description="Información sobre manuales de software y operativas habituales dentro de la empresa.",
                retrieval_type="semantic",
                collection_name="manuals",
                dense_vector_name=None,
                sparse_vector_name=None,
            ),
            "tickets": KnowledgeSourceDefinition(
                id="tickets",
                name="Tickets",
                description="Información sobre los tickets registrados en Helpdesk.",
                retrieval_type="hybrid",
                collection_name="tickets",
                dense_vector_name="dense_vector",
                sparse_vector_name="sparse_vector",
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
    
