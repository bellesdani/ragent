from app.core.entities import KnowledgeSource


class QdrantKnowledgeSourceCatalog():
    def __init__(self) -> None:
        self._knowledge_sources = [
            KnowledgeSource(
                id="devices",
                name="Devices",
                description="Información sobre dispositivos de la empresa, como servidores y equipos de usuario y planta.",
                collection="devices",
            ),
            KnowledgeSource(
                id="employees",
                name="Employees",
                description="Información sobre empleados y su contacto corporativo: correo, teléfono y extensión.",
                collection="employees",
            ),
            KnowledgeSource(
                id="manuals",
                name="Manuals",
                description="Información sobre manuales de software y operativas habituales dentro de la empresa.",
                collection="manuals",
            ),
            KnowledgeSource(
                id="tickets",
                name="Tickets",
                description="Información sobre los tickets registrados en Helpdesk.",
                collection="tickets",
            ),
        ]


    def get_knowledge_sources(self) -> list[KnowledgeSource]:
        return self._knowledge_sources


    def get_knowledge_sources_by_id(self) -> dict[str, KnowledgeSource]:
        return {
            source.id: source 
            for source in self._knowledge_sources
        }
    

