from app.config import Settings
from app.core.prompts import load_prompt
from app.core.entities import AgentDefinition


class AgentCatalog:

    def __init__(self, settings: Settings) -> None:
        self._agents = {
            "Quipi": AgentDefinition(
                agent_id="Quipi",
                name="Quipi",
                description="Agente conversacional corporativo con acceso a herramientas de búsqueda sobre Qdrant.",
                backend_base_url=settings.chat_base_url,
                backend_api_key=settings.chat_api_key,
                backend_chat_model=settings.chat_model,
                system_prompt=load_prompt("quipi_system.md"),
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
                system_prompt=load_prompt("base_system.md"),
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
                system_prompt=load_prompt("summarizer_system.md"),
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