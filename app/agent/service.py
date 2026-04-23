from __future__ import annotations

import json
import uuid

from app.core.config import Settings
from collections.abc import AsyncIterator
from app.agent.catalog import AgentCatalog
from app.retrieval.base import BaseRetriever
from app.llm.openai_compat import OpenAICompatClient
from app.api.schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChunk,
    ChatCompletionStreamChoice,
    ChatCompletionDelta,
    ChatMessage,
    RetrievedContext,
)


class ChatAgentService:
    def __init__(
        self,
        settings: Settings,
        retriever: BaseRetriever,
        llm_client: OpenAICompatClient,
        agent_catalog: AgentCatalog,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.llm_client = llm_client
        self.agent_catalog = agent_catalog

    async def create_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        agent = self.agent_catalog.get_agent(request.model)
        retrieval = await self._resolve_retrieval(agent, request.messages)
        messages = self._build_llm_messages(
            request.messages,
            retrieval.documents,
            agent.system_prompt,
            agent.use_retrieval,
        )
        llm_result = await self.llm_client.create_chat_completion(
            model=agent.backend_chat_model,
            messages=messages,
            temperature=request.temperature or self.settings.llm_temperature,
            max_tokens=request.max_tokens or self.settings.llm_max_tokens,
        )
        return ChatCompletionResponse(
            model=agent.agent_id,
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionChoiceMessage(content=llm_result.content),
                )
            ],
            usage=llm_result.usage,
        )

    async def _resolve_retrieval(self, agent, messages: list[ChatMessage]) -> RetrievedContext:
        query = self._latest_user_message(messages)
        if not agent.use_retrieval:
            return RetrievedContext(query=query, documents=[])
        return await self.retriever.retrieve(query, messages)

    async def stream_chat_completion(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        response = await self.create_chat_completion(request)
        chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
        first_chunk = ChatCompletionStreamChunk(
            id=chunk_id,
            model=response.model,
            choices=[ChatCompletionStreamChoice(delta=ChatCompletionDelta(role="assistant"))],
        )
        yield self._format_sse(first_chunk.model_dump())
        content_chunk = ChatCompletionStreamChunk(
            id=chunk_id,
            model=response.model,
            choices=[ChatCompletionStreamChoice(delta=ChatCompletionDelta(content=response.choices[0].message.content))],
        )
        yield self._format_sse(content_chunk.model_dump())
        final_chunk = ChatCompletionStreamChunk(
            id=chunk_id,
            model=response.model,
            choices=[ChatCompletionStreamChoice(delta=ChatCompletionDelta(), finish_reason="stop")],
        )
        yield self._format_sse(final_chunk.model_dump())
        yield "data: [DONE]\n\n"

    def _latest_user_message(self, messages: list[ChatMessage]) -> str:
        for message in reversed(messages):
            if message.role == "user" and message.content:
                return message.content
        raise ValueError("user message content is required")

    def _build_llm_messages(
        self,
        messages: list[ChatMessage],
        documents: list,
        system_prompt: str,
        use_retrieval: bool,
    ) -> list[dict[str, str]]:
        history = [message for message in messages if message.role in {"user", "assistant", "system"} and message.content]
        system_parts = [system_prompt]
        if use_retrieval and documents:
            citations = []
            for index, doc in enumerate(documents, start=1):
                source = doc.metadata.get("source") or doc.metadata.get("title") or doc.id
                citations.append(f"[{index}] {source}\n{doc.text}")
            system_parts.append("Contexto recuperado:\n" + "\n\n".join(citations))
        elif use_retrieval:
            system_parts.append("No se ha recuperado contexto relevante.")

        llm_messages = [{"role": "system", "content": "\n\n".join(system_parts)}]
        llm_messages.extend({"role": message.role, "content": message.content} for message in history if message.role != "system")
        return llm_messages

    def _format_sse(self, payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
