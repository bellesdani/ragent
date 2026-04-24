from __future__ import annotations

import json
import uuid
import logging
from typing import Any
from app.config.config import Settings
from collections.abc import AsyncIterator
from app.agent.catalog import AgentCatalog
from app.retrieval.base import BaseRetriever
from app.agent.runtime import PydanticAgentRuntime
from app.api.schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChunk,
    ChatCompletionStreamChoice,
    ChatMessage,
)


logger = logging.getLogger(__name__)


class ChatAgentService:
    def __init__(self, settings: Settings, retriever: BaseRetriever, agent_catalog: AgentCatalog) -> None:
        self.settings = settings
        self.retriever = retriever
        self.agent_catalog = agent_catalog
        self.agent_runtime = PydanticAgentRuntime(settings=settings, retriever=retriever)

    async def create_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        agent = self.agent_catalog.get_agent(request.model)
        logger.debug(
            "Received chat completion request | agent=%s request_model=%s stream=%s messages=%d latest_user=%s",
            agent.agent_id,
            request.model,
            request.stream,
            len(request.messages),
            self._safe_latest_user_message(request.messages),
        )
        content, usage = await self.agent_runtime.run(
            definition=agent,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
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

    # async def stream_chat_completion(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
    #     agent = self.agent_catalog.get_agent(request.model)
    #     chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    #     logger.debug("Starting streaming response | agent=%s", agent.agent_id)

    #     # Inicio del streaming
    #     yield self._format_sse(
    #         ChatCompletionStreamChunk(
    #             id=chunk_id,
    #             model=agent.agent_id,
    #             choices=[ChatCompletionStreamChoice(delta=ChatCompletionDelta(role="assistant"))],
    #         ).model_dump()
    #     )

    #     # Generación del contenido
    #     content, _usage = await self.agent_runtime.run(
    #         definition=agent,
    #         messages=request.messages,
    #         temperature=request.temperature,
    #         max_tokens=request.max_tokens,
    #     )

    #     # Envio del contenido 
    #     if content:
    #         yield self._format_sse(
    #             ChatCompletionStreamChunk(
    #                 id=chunk_id,
    #                 model=agent.agent_id,
    #                 choices=[ChatCompletionStreamChoice(delta=ChatCompletionDelta(content=content))],
    #             ).model_dump()
    #         )

    #     # Finalización del streaming
    #     yield self._format_sse(
    #         ChatCompletionStreamChunk(
    #             id=chunk_id,
    #             model=agent.agent_id,
    #             choices=[ChatCompletionStreamChoice(delta=ChatCompletionDelta(), finish_reason="stop")],
    #         ).model_dump()
    #     )
    #     yield "data: [DONE]\n\n"

    def _safe_latest_user_message(self, messages: list[ChatMessage]) -> str | None:
        for message in reversed(messages):
            if message.role == "user" and message.content:
                return message.content
        return None

