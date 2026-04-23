from __future__ import annotations

import uuid
import json
import logging

from typing import Any
from app.config.config import Settings
from collections.abc import AsyncIterator
from app.retrieval.base import BaseRetriever
from app.llm.openai_compat import OpenAICompatClient
from app.agent.catalog import AgentCatalog, AgentDefinition
from app.agent.planner import PlannerDecision, RetrievalPlanner
from app.api.schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChunk,
    ChatCompletionStreamChoice,
    ChatCompletionUsage,
    ChatMessage,
    RetrievalDocument,
)


logger = logging.getLogger(__name__)


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
        self.retrieval_planner = RetrievalPlanner(llm_client=llm_client, retriever=retriever, settings=settings)

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
        llm_result = await self._run_agent(agent, request)
        logger.debug(
            "Returning chat completion response | agent=%s has_content=%s preview=%s",
            agent.agent_id,
            bool(llm_result.content and llm_result.content.strip()),
            (llm_result.content or "")[:300],
        )
        return ChatCompletionResponse(
            model=agent.agent_id,
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionChoiceMessage(content=llm_result.content or ""),
                )
            ],
            usage=llm_result.usage,
        )

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

    async def _run_agent(self, agent: AgentDefinition, request: ChatCompletionRequest):
        if not agent.use_planner:
            logger.debug("Running agent without planner | agent=%s", agent.agent_id)
            return await self._run_direct_completion(
                agent=agent,
                messages=self._build_base_messages(request.messages, agent.system_prompt),
                request=request,
            )

        logger.debug("Running agent with retrieval planner | agent=%s", agent.agent_id)
        plan = await self.retrieval_planner.plan(
            agent=agent,
            messages=request.messages,
        )
        logger.debug(
            "Planner decision | agent=%s should_search=%s query=%s sources=%s reason=%s",
            agent.agent_id,
            plan.should_search,
            plan.query,
            plan.sources,
            plan.reason,
        )

        documents: list[RetrievalDocument] = []
        accumulated_usage = plan.usage
        if plan.should_search:
            retrieval = await self.retriever.retrieve_from_sources(
                query=plan.query or self._latest_user_message(request.messages),
                messages=request.messages,
                source_ids=plan.sources or None,
            )
            documents = retrieval.documents
            logger.debug(
                "Planner-triggered retrieval completed | agent=%s documents=%d query=%s",
                agent.agent_id,
                len(documents),
                retrieval.query,
            )

        llm_result = await self._run_direct_completion(
            agent=agent,
            messages=self._build_answer_messages(
                messages=request.messages,
                system_prompt=agent.system_prompt,
                documents=documents,
                plan=plan,
            ),
            request=request,
        )
        llm_result.usage = self._merge_usage(accumulated_usage, llm_result.usage)
        return llm_result

    async def _run_direct_completion(self, agent: AgentDefinition, messages: list[dict[str, Any]], request: ChatCompletionRequest):
        logger.debug("Running direct completion | agent=%s messages=%d", agent.agent_id, len(messages))
        return await self.llm_client.create_chat_completion(
            model=agent.backend_chat_model,
            messages=messages,
            temperature=request.temperature or self.settings.llm_temperature,
            max_tokens=request.max_tokens or self.settings.llm_max_tokens,
        )

    def _build_base_messages(self, messages: list[ChatMessage], system_prompt: str) -> list[dict[str, Any]]:
        history = [
            message
            for message in messages
            if message.role in {"user", "assistant", "system"} and message.content
        ]
        llm_messages = [{"role": "system", "content": system_prompt}]
        llm_messages.extend(
            {"role": message.role, "content": message.content}
            for message in history
            if message.role != "system"
        )
        return llm_messages

    def _build_answer_messages(
        self,
        messages: list[ChatMessage],
        system_prompt: str,
        documents: list[RetrievalDocument],
        plan: PlannerDecision,
    ) -> list[dict[str, Any]]:
        system_parts = [system_prompt]
        if plan.should_search:
            if documents:
                citations = []
                for index, document in enumerate(documents, start=1):
                    source = document.metadata.get("source_name") or document.metadata.get("collection") or document.id
                    citations.append(f"[{index}] {source}\n{document.text}")
                system_parts.append("Contexto recuperado:\n" + "\n\n".join(citations))
            else:
                system_parts.append(
                    "Se intentó buscar contexto relevante, pero no se encontró evidencia suficiente en las fuentes seleccionadas."
                )
        else:
            system_parts.append(
                "No es necesario consultar conocimiento externo para esta respuesta. Responde directamente usando solo el historial."
            )

        history = [
            message
            for message in messages
            if message.role in {"user", "assistant", "system"} and message.content
        ]
        llm_messages = [{"role": "system", "content": "\n\n".join(system_parts)}]
        llm_messages.extend(
            {"role": message.role, "content": message.content}
            for message in history
            if message.role != "system"
        )
        return llm_messages

    def _latest_user_message(self, messages: list[ChatMessage]) -> str:
        for message in reversed(messages):
            if message.role == "user" and message.content:
                return message.content
        raise ValueError("user message content is required")

    def _safe_latest_user_message(self, messages: list[ChatMessage]) -> str | None:
        try:
            return self._latest_user_message(messages)
        except ValueError:
            return None

    def _merge_usage(self, left: ChatCompletionUsage, right: ChatCompletionUsage) -> ChatCompletionUsage:
        return ChatCompletionUsage(
            prompt_tokens=left.prompt_tokens + right.prompt_tokens,
            completion_tokens=left.completion_tokens + right.completion_tokens,
            total_tokens=left.total_tokens + right.total_tokens,
        )

    def _format_sse(self, payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
