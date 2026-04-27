from __future__ import annotations

from app.retrieval.qdrant import QdrantRetriever
from app.agent.registry import AgentCatalog
from app.agent.runtime import AgentRunner
from app.config.config import Settings
from app.api.schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
)


class ChatAgentService:
    def __init__(self, settings: Settings, retriever: QdrantRetriever, agent_catalog: AgentCatalog) -> None:
        self.settings = settings
        self.retriever = retriever
        self.agent_catalog = agent_catalog
        self.agent_runtime = AgentRunner(settings=settings, retriever=retriever)

    async def create_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        # Get selected agent
        agent = self.agent_catalog.get_agent(request.model)
        # Run pydanticAI agent
        content, usage = await self.agent_runtime.run(
            definition=agent,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        # Return agent response
        return ChatCompletionResponse(
            model=agent.agent_id,
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionChoiceMessage(content=content),
                )
            ],
            usage=usage,
        )
