from __future__ import annotations

from dataclasses import dataclass
from app.config.config import Settings


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
        self._agents = {
            "Quipi": AgentDefinition(
                agent_id="Quipi",
                name="Quipi",
                description="Agente conversacional corporativo con acceso a herramientas de búsqueda sobre Qdrant.",
                backend_chat_model=settings.chat_model,
                system_prompt=(
                    "Eres Quipi, el agente corporativo de Equipe Cerámicas. "
                    "Responde en español, con precisión y con un tono profesional. "
                    "Puedes apoyarte en conocimiento corporativo cuando la situación lo requiera."
                ),
                use_planner=True,
            ),
            "Base": AgentDefinition(
                agent_id="Base",
                name="Base",
                description="Agente conversacional básico sin conexiones a herramientas de búsqueda.",
                backend_chat_model=settings.chat_model,
                system_prompt=(
                    "Eres Quipi, el agente corporativo de Equipe Cerámicas. "
                    "Responde de forma útil, breve y natural."
                ),
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
