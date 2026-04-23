from __future__ import annotations

import json
import logging
import uuid

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from app.agent.catalog import AgentCatalog, AgentDefinition
from app.agent.planner import PlannerDecision, RetrievalPlanner
from app.agent.prompt_builder import PromptBuilder
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
from app.config.config import Settings
from app.llm.openai_compat import OpenAICompatClient
from app.retrieval.base import BaseRetriever


logger = logging.getLogger(__name__)


@dataclass
class PreparedAgentCompletion:
    agent: AgentDefinition
    messages: list[dict[str, Any]]
    accumulated_usage: ChatCompletionUsage


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
        self.prompt_builder = PromptBuilder()
        self.retrieval_planner = RetrievalPlanner(llm_client=llm_client, retriever=retriever)

    async def create_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        prepared = await self._prepare_agent_completion(request)
        llm_result = await self._run_direct_completion(
            agent=prepared.agent,
            messages=prepared.messages,
            request=request,
        )
        llm_result.usage = self._merge_usage(prepared.accumulated_usage, llm_result.usage)
        logger.debug(
            "Returning chat completion response | agent=%s has_content=%s preview=%s",
            prepared.agent.agent_id,
            bool(llm_result.content and llm_result.content.strip()),
            (llm_result.content or "")[:300],
        )
        return ChatCompletionResponse(
            model=prepared.agent.agent_id,
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionChoiceMessage(content=llm_result.content or ""),
                )
            ],
            usage=llm_result.usage,
        )

    async def stream_chat_completion(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        prepared = await self._prepare_agent_completion(request)
        chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
        logger.debug("Starting streaming response | agent=%s", prepared.agent.agent_id)

        yield self._format_sse(
            ChatCompletionStreamChunk(
                id=chunk_id,
                model=prepared.agent.agent_id,
                choices=[ChatCompletionStreamChoice(delta=ChatCompletionDelta(role="assistant"))],
            ).model_dump()
        )

        async for content in self.llm_client.stream_chat_completion(
            model=prepared.agent.backend_chat_model,
            messages=prepared.messages,
            temperature=request.temperature or self.settings.llm_temperature,
            max_tokens=request.max_tokens or self.settings.llm_max_tokens,
        ):
            yield self._format_sse(
                ChatCompletionStreamChunk(
                    id=chunk_id,
                    model=prepared.agent.agent_id,
                    choices=[ChatCompletionStreamChoice(delta=ChatCompletionDelta(content=content))],
                ).model_dump()
            )

        yield self._format_sse(
            ChatCompletionStreamChunk(
                id=chunk_id,
                model=prepared.agent.agent_id,
                choices=[ChatCompletionStreamChoice(delta=ChatCompletionDelta(), finish_reason="stop")],
            ).model_dump()
        )
        yield "data: [DONE]\n\n"

    async def _prepare_agent_completion(self, request: ChatCompletionRequest) -> PreparedAgentCompletion:
        agent = self.agent_catalog.get_agent(request.model)
        logger.debug(
            "Received chat completion request | agent=%s request_model=%s stream=%s messages=%d latest_user=%s",
            agent.agent_id,
            request.model,
            request.stream,
            len(request.messages),
            self._safe_latest_user_message(request.messages),
        )

        if not agent.use_planner:
            logger.debug("Running agent without planner | agent=%s", agent.agent_id)
            return PreparedAgentCompletion(
                agent=agent,
                messages=self._build_base_messages(request.messages, agent.system_prompt),
                accumulated_usage=self._empty_usage(),
            )

        logger.debug("Running agent with retrieval planner | agent=%s", agent.agent_id)
        plan = await self.retrieval_planner.plan(agent=agent, messages=request.messages)
        logger.debug(
            "Planner decision | agent=%s should_search=%s query=%s sources=%s reason=%s",
            agent.agent_id,
            plan.should_search,
            plan.query,
            plan.sources,
            plan.reason,
        )

        documents: list[RetrievalDocument] = []
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

        return PreparedAgentCompletion(
            agent=agent,
            messages=self._build_answer_messages(
                messages=request.messages,
                system_prompt=agent.system_prompt,
                documents=documents,
                plan=plan,
            ),
            accumulated_usage=plan.usage,
        )

    async def _run_direct_completion(
        self,
        agent: AgentDefinition,
        messages: list[dict[str, Any]],
        request: ChatCompletionRequest,
    ):
        logger.debug("Running direct completion | agent=%s messages=%d", agent.agent_id, len(messages))
        return await self.llm_client.create_chat_completion(
            model=agent.backend_chat_model,
            messages=messages,
            temperature=request.temperature or self.settings.llm_temperature,
            max_tokens=request.max_tokens or self.settings.llm_max_tokens,
        )

    def _build_base_messages(self, messages: list[ChatMessage], system_prompt: str) -> list[dict[str, Any]]:
        return self._compose_llm_messages(messages, system_prompt)

    def _build_answer_messages(
        self,
        messages: list[ChatMessage],
        system_prompt: str,
        documents: list[RetrievalDocument],
        plan: PlannerDecision,
    ) -> list[dict[str, Any]]:
        answer_system_prompt = self.prompt_builder.build_answer_system_prompt(
            base_system_prompt=system_prompt,
            documents=documents,
            should_search=plan.should_search,
            format_metadata=self._format_document_metadata,
        )
        return self._compose_llm_messages(messages, answer_system_prompt)

    def _compose_llm_messages(self, messages: list[ChatMessage], system_prompt: str) -> list[dict[str, Any]]:
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

    def _empty_usage(self) -> ChatCompletionUsage:
        return ChatCompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    def _merge_usage(self, left: ChatCompletionUsage, right: ChatCompletionUsage) -> ChatCompletionUsage:
        return ChatCompletionUsage(
            prompt_tokens=left.prompt_tokens + right.prompt_tokens,
            completion_tokens=left.completion_tokens + right.completion_tokens,
            total_tokens=left.total_tokens + right.total_tokens,
        )

    def _format_document_metadata(self, metadata: dict[str, Any]) -> str:
        if not metadata:
            return ""
        visible_metadata = {
            key: value
            for key, value in metadata.items()
            if value not in (None, "", [], {})
        }
        if not visible_metadata:
            return ""
        return json.dumps(visible_metadata, ensure_ascii=False, default=str)

    def _format_sse(self, payload: dict[str, Any]) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
