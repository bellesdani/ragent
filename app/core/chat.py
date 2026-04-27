from __future__ import annotations

from app.core.agent_runner import AgentRunner
from app.core.config import Settings
from app.core.openai import OpenAICompatClient
from app.core.retrieval import QdrantRetriever
from app.core.agent_catalog import AgentCatalog

from app.core.entities import AgentDefinition, ChatMessage, ChatResult


class ChatAgentService:
    def __init__(self, settings: Settings, retriever: QdrantRetriever, agent_catalog: AgentCatalog) -> None:
        self.settings = settings
        self.retriever = retriever
        self.agent_catalog = agent_catalog
        self.agent_runtime = AgentRunner(settings=settings, retriever=retriever)

    def list_agents(self) -> list[AgentDefinition]:
        return self.agent_catalog.list_agents()

    async def complete(self, model: str, messages: list[ChatMessage], temperature: float | None, max_tokens: int | None) -> ChatResult:
        agent = self.agent_catalog.get_agent(model)
        content, usage = await self.agent_runtime.run(
            definition=agent,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return ChatResult(model=agent.agent_id, content=content, usage=usage)


def build_chat_service(settings: Settings) -> ChatAgentService:
    agent_catalog = AgentCatalog(settings)
    embedding_client = OpenAICompatClient(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        timeout=settings.llm_timeout_seconds,
    )
    retriever = QdrantRetriever(settings=settings, embedding_client=embedding_client)
    return ChatAgentService(
        settings=settings,
        retriever=retriever,
        agent_catalog=agent_catalog,
    )
