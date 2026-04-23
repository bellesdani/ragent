from __future__ import annotations

from dataclasses import dataclass
from app.config.config import Settings
from app.agent.prompt_builder import PromptBuilder


@dataclass(frozen=True)
class AgentDefinition:
    agent_id: str
    name: str
    description: str
    backend_chat_model: str
    system_prompt: str
    use_planner: bool


class AgentCatalog:
    def __init__(self, settings: Settings) -> None:
        prompt_builder = PromptBuilder()
        self._agents = {
            "Quipi": AgentDefinition(
                agent_id="Quipi",
                name="Quipi",
                description="Agente conversacional corporativo con acceso a herramientas de busqueda sobre Qdrant.",
                backend_chat_model=settings.chat_model,
                system_prompt=prompt_builder.build_agent_system_prompt("agents/quipi_system.md"),
                use_planner=True,
            ),
            "Base": AgentDefinition(
                agent_id="Base",
                name="Base",
                description="Agente conversacional basico sin conexiones a herramientas de busqueda.",
                backend_chat_model=settings.chat_model,
                system_prompt=prompt_builder.build_agent_system_prompt("agents/base_system.md"),
                use_planner=False,
            ),
        }

    def list_agents(self) -> list[AgentDefinition]:
        return list(self._agents.values())

    def get_agent(self, agent_id: str | None) -> AgentDefinition:
        if not agent_id:
            return self._agents["Quipi"]
        if agent_id in self._agents:
            return self._agents[agent_id]
        available = ", ".join(sorted(self._agents))
        raise ValueError(f"Unknown agent '{agent_id}'. Available agents: {available}")
