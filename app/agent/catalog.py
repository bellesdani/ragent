from __future__ import annotations

from dataclasses import dataclass
from app.config.config import Settings
from app.agent.prompt_builder import PromptBuilder


@dataclass(frozen=True)
class AgentDefinition:
    agent_id: str
    name: str
    description: str
    backend_base_url: str
    backend_api_key: str
    backend_chat_model: str
    system_prompt: str
    enable_retrieval: bool


class AgentCatalog:
    def __init__(self, settings: Settings) -> None:
        prompt_builder = PromptBuilder()
        self._agents = {
            "Quipi": AgentDefinition(
                agent_id="Quipi",
                name="Quipi",
                description="Agente conversacional corporativo con acceso a herramientas de busqueda sobre Qdrant.",
                backend_base_url=settings.chat_base_url,
                backend_api_key=settings.chat_api_key,
                backend_chat_model=settings.chat_model,
                system_prompt=prompt_builder.build_agent_system_prompt("agents/quipi_system.md"),
                enable_retrieval=True,
            ),
            "Base": AgentDefinition(
                agent_id="Base",
                name="Base",
                description="Agente conversacional basico sin conexiones a herramientas de busqueda.",
                backend_base_url=settings.chat_base_url,
                backend_api_key=settings.chat_api_key,
                backend_chat_model=settings.chat_model,
                system_prompt=prompt_builder.build_agent_system_prompt("agents/base_system.md"),
                enable_retrieval=False,
            ),
        }

    def list_agents(self) -> list[AgentDefinition]:
        return list(self._agents.values())

    def get_agent(self, agent_id: str) -> AgentDefinition:
        if agent_id in self._agents:
            return self._agents[agent_id]
        raise ValueError(f"Unknown agent '{agent_id}'. Available agents: {", ".join(sorted(self._agents))}")
