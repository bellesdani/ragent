from __future__ import annotations

from app.agent.deps import AgentDeps
from app.config.config import Settings
from pydantic_ai.usage import UsageLimits
from pydantic_ai.settings import ModelSettings
from app.retrieval.qdrant import QdrantRetriever
from app.agent.definitions import AgentDefinition
from app.agent.runtime.factory import AgentFactory
from app.agent.runtime.messages import split_messages
from app.api.schemas.openai import ChatCompletionUsage, ChatMessage


class AgentRunner:
    def __init__(self, settings: Settings, retriever: QdrantRetriever) -> None:
        self.settings = settings
        self.retriever = retriever
        self.factory = AgentFactory(settings)

    async def run(
        self,
        definition: AgentDefinition,
        messages: list[ChatMessage],
        temperature: float | None,
        max_tokens: int | None,
    ) -> tuple[str, ChatCompletionUsage]:
        agent = self.factory.build(definition)
        latest_user_prompt, message_history = split_messages(messages, definition.system_prompt)
        result = await agent.run(
            latest_user_prompt,
            deps=AgentDeps(retriever=self.retriever, messages=messages),
            message_history=message_history,
            model_settings=ModelSettings(
                temperature=temperature if temperature is not None else self.settings.llm_temperature,
                max_tokens=max_tokens if max_tokens is not None else self.settings.llm_max_tokens,
            ),
            usage_limits=self._build_usage_limits(),
        )
        usage = result.usage()
        return result.output, ChatCompletionUsage(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        )

    def _build_usage_limits(self) -> UsageLimits:
        return UsageLimits(request_limit=10)
