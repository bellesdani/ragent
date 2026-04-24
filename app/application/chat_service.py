from __future__ import annotations

import logging

from app.agent.registry import AgentCatalog
from app.agent.runtime import AgentRunner
from app.api.schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)
from app.config.config import Settings
from app.retrieval.base import BaseRetriever


logger = logging.getLogger(__name__)


class ChatAgentService:
    def __init__(self, settings: Settings, retriever: BaseRetriever, agent_catalog: AgentCatalog) -> None:
        self.settings = settings
        self.retriever = retriever
        self.agent_catalog = agent_catalog
        self.agent_runtime = AgentRunner(settings=settings, retriever=retriever)

    async def create_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        agent = self.agent_catalog.get_agent(request.model)
        logger.debug(
            "Received chat completion request | agent=%s request_model=%s stream=%s messages=%d latest_user=%s",
            agent.agent_id,
            request.model,
            request.stream,
            len(request.messages),
            self._latest_user_message(request.messages),
        )
        content, usage = await self.agent_runtime.run(
            definition=agent,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        logger.debug(
            "Finished chat completion request | agent=%s request_model=%s stream=%s messages=%d latest_user=%s",
            agent.agent_id,
            request.model,
            request.stream,
            len(request.messages),
            self._latest_user_message(request.messages),
        )
        return ChatCompletionResponse(
            model=agent.agent_id,
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionChoiceMessage(content=content),
                )
            ],
            usage=usage,
        )

    def _latest_user_message(self, messages: list[ChatMessage]) -> str | None:
        for message in reversed(messages):
            if message.role == "user" and message.content:
                return message.content
        return None
