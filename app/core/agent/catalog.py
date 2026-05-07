from app.config import Settings
from app.core.prompts import PromptService
from app.core.entities import AgentDefinition


class AgentCatalog:
    """
    Este catálogo centraliza las definiciones de agentes disponibles. Utiliza:
     - Las variables cargadas (Settings)
     - El servicio de prompts (PromptService)

    Funciones públicas:
     - Listar los agentes publicados (list_agents).
     - Obtener un agente por identificador (get_agent).
    """

    def __init__(self, settings: Settings, prompt_service: PromptService) -> None:
        self._agents = {
            "Quipi": AgentDefinition(
                agent_id="Quipi",
                name="Quipi",
                description="Agente conversacional corporativo con acceso a herramientas de búsqueda sobre Qdrant.",
                backend_base_url=settings.chat_base_url,
                backend_api_key=settings.chat_api_key,
                backend_chat_model=settings.chat_model,
                system_prompt=prompt_service.load_prompt("quipi_system.md"),
                enable_tools=True,
                public=True,
            ),
            "Base": AgentDefinition(
                agent_id="Base",
                name="Base",
                description="Agente conversacional básico sin conexiones a herramientas de búsqueda.",
                backend_base_url=settings.chat_base_url,
                backend_api_key=settings.chat_api_key,
                backend_chat_model=settings.chat_model,
                system_prompt=prompt_service.load_prompt("base_system.md"),
                enable_tools=False,
                public=False,
            ),
            "Summarizer": AgentDefinition(
                agent_id="Summarizer",
                name="Summarizer",
                description="Agente especializado en resumir las incidencias de HelpDesk.",
                backend_base_url=settings.chat_base_url,
                backend_api_key=settings.chat_api_key,
                backend_chat_model=settings.chat_model,
                system_prompt=prompt_service.load_prompt("summarizer_system.md"),
                enable_tools=False,
                public=False,
            ),
        }


    def list_agents(self) -> list[AgentDefinition]:
        return [
            agent  
            for agent in self._agents.values()
            if agent.public
        ]


    def get_agent(self, agent_id: str) -> AgentDefinition:
        if agent_id in self._agents:
            return self._agents[agent_id]
        raise ValueError(f"Agente '{agent_id}' desconocido. Agentes disponibles: {', '.join(sorted(self._agents))}")
