from __future__ import annotations

import json
import logging

from dataclasses import dataclass
from typing import Any

from app.retrieval.base import BaseRetriever
from app.agent.catalog import AgentDefinition
from app.agent.prompt_builder import PromptBuilder
from app.llm.openai_compat import OpenAICompatClient
from app.api.schemas.openai import ChatCompletionUsage, ChatMessage


logger = logging.getLogger(__name__)


@dataclass
class PlannerDecision:
    should_search: bool
    query: str | None
    sources: list[str]
    reason: str
    usage: ChatCompletionUsage


class RetrievalPlanner:
    def __init__(self, llm_client: OpenAICompatClient, retriever: BaseRetriever) -> None:
        self.llm_client = llm_client
        self.retriever = retriever
        self.prompt_builder = PromptBuilder()

    async def plan(self, agent: AgentDefinition, messages: list[ChatMessage]) -> PlannerDecision:
        planner_messages = self._build_planner_messages(agent, messages)
        logger.debug(
            "Running retrieval planner | agent=%s sources=%s latest_user=%s",
            agent.agent_id,
            self.retriever.list_sources(),
            self._latest_user_message(messages),
        )
        llm_result = await self.llm_client.create_chat_completion(
            model=agent.backend_chat_model,
            messages=planner_messages,
            temperature=0.0,
            max_tokens=250,
        )
        raw_content = llm_result.content or ""
        logger.debug("Planner raw response | agent=%s content=%s", agent.agent_id, raw_content)
        parsed = self._parse_planner_output(raw_content)
        return PlannerDecision(
            should_search=bool(parsed.get("should_search", False)),
            query=self._normalize_query(parsed.get("query"), messages),
            sources=self._normalize_sources(parsed.get("sources")),
            reason=str(parsed.get("reason") or ""),
            usage=llm_result.usage,
        )

    def _build_planner_messages(self, agent: AgentDefinition, messages: list[ChatMessage]) -> list[dict[str, str]]:
        sources = self.retriever.list_sources()
        history_lines = []
        for message in messages[-6:]:
            if message.content and message.role in {"user", "assistant"}:
                history_lines.append(f"{message.role}: {message.content}")
        planner_prompt = self.prompt_builder.build_planner_prompt(agent_id=agent.agent_id, sources=sources)
        return [
            {"role": "system", "content": planner_prompt},
            {"role": "user", "content": "\n".join(history_lines) or self._latest_user_message(messages)},
        ]

    def _parse_planner_output(self, content: str) -> dict[str, Any]:
        stripped = content.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3:
                stripped = "\n".join(lines[1:-1]).strip()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            logger.warning("Planner returned invalid JSON, falling back to direct response | raw=%s", content)
            return {"should_search": False, "query": None, "sources": [], "reason": "invalid_json"}
        if not isinstance(parsed, dict):
            logger.warning("Planner returned non-object JSON, falling back to direct response | raw=%s", content)
            return {"should_search": False, "query": None, "sources": [], "reason": "invalid_shape"}
        return parsed

    def _normalize_query(self, query: Any, messages: list[ChatMessage]) -> str:
        if isinstance(query, str) and query.strip():
            return query.strip()
        return self._latest_user_message(messages)

    def _normalize_sources(self, sources: Any) -> list[str]:
        available_sources = {source["id"] for source in self.retriever.list_sources()}
        if not isinstance(sources, list):
            return []
        return [source for source in sources if isinstance(source, str) and source in available_sources]

    def _latest_user_message(self, messages: list[ChatMessage]) -> str:
        for message in reversed(messages):
            if message.role == "user" and message.content:
                return message.content
        return ""
