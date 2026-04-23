from __future__ import annotations
from dataclasses import dataclass
from app.core.config import Settings


@dataclass(frozen=True)
class AgentDefinition:
    agent_id: str
    description: str
    backend_chat_model: str
    system_prompt: str
    use_retrieval: bool


class AgentCatalog:
    def __init__(self, settings: Settings) -> None:
        self._agents = {
            "Quipi": AgentDefinition(
                agent_id="Quipi",
                description="Agente conversacional corporativo con RAG sobre Qdrant y generación con vLLM.",
                backend_chat_model=settings.chat_model,
                system_prompt=(
                    "Eres Quipi, el agente corporativo de Equipe Cerámicas con acceso al conocmiento corporativo de la empresa. "
                ),
                use_retrieval=True,
            ),
            "Base": AgentDefinition(
                agent_id="Base",
                description="Agente conversacional básico sin conexiones a herramientas de búsqueda.",
                backend_chat_model=settings.chat_model,
                system_prompt=(
                    "Eres Quipi, el agente corporativo de Equipe Cerámicas. "
                ),
                use_retrieval=False,
            ),
        }

    def list_agents(self) -> list[AgentDefinition]:
        return list(self._agents.values())

    def get_agent(self, agent_id: str | None) -> AgentDefinition:
        if not agent_id:
            return self._agents["quipi"]
        if agent_id in self._agents:
            return self._agents[agent_id]
        available = ", ".join(sorted(self._agents))
        raise ValueError(f"Unknown agent '{agent_id}'. Available agents: {available}")
