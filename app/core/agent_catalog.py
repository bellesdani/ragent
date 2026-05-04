from app.core.config import Settings
from app.core.prompts import load_prompt
from app.core.entities import AgentDefinition


class AgentCatalog:

    def __init__(self, settings: Settings) -> None:
        self._agents = {
            "Quipi": AgentDefinition(
                agent_id="Quipi",
                name="Quipi",
                description="Agente conversacional corporativo con acceso a herramientas de busqueda sobre Qdrant.",
                backend_base_url=settings.chat_base_url,
                backend_api_key=settings.chat_api_key,
                backend_chat_model=settings.chat_model,
                system_prompt=load_prompt("quipi_system.md"),
                enable_tools=True,
            ),
            # "Base": AgentDefinition(
            #     agent_id="Base",
            #     name="Base",
            #     description="Agente conversacional basico sin conexiones a herramientas de busqueda.",
            #     backend_base_url=settings.chat_base_url,
            #     backend_api_key=settings.chat_api_key,
            #     backend_chat_model=settings.chat_model,
            #     system_prompt=load_prompt("base_system.md"),
            #     enable_tools=False,
            # ),
        }


    def list_agents(self) -> list[AgentDefinition]:
        return list(self._agents.values())


    def get_agent(self, agent_id: str) -> AgentDefinition:
        if agent_id in self._agents:
            return self._agents[agent_id]
        raise ValueError(f"Unknown agent '{agent_id}'. Available agents: {', '.join(sorted(self._agents))}")