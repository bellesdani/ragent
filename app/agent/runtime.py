from __future__ import annotations

from pydantic_ai import Agent
from openai import AsyncOpenAI
from app.agent.types import AgentDeps
from app.config.config import Settings
from pydantic_ai.usage import UsageLimits
from app.retrieval.base import BaseRetriever
from app.agent.catalog import AgentDefinition
from pydantic_ai.settings import ModelSettings
from app.agent.tools import register_agent_tools
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from app.api.schemas.openai import ChatMessage, ChatCompletionUsage
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart


class PydanticAgentRuntime:
    def __init__(self, settings: Settings, retriever: BaseRetriever) -> None:
        self.settings = settings
        self.retriever = retriever

    def build_agent(self, definition: AgentDefinition) -> Agent[AgentDeps, str]:
        provider = OpenAIProvider(
            openai_client=AsyncOpenAI(
                base_url=definition.backend_base_url.rstrip("/") + "/",
                api_key=definition.backend_api_key or "api-key-not-set",
                timeout=self.settings.llm_timeout_seconds,
            )
        )
        model = OpenAIChatModel(
            definition.backend_chat_model,
            provider=provider,
            settings=ModelSettings(
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_tokens,
            ),
        )
        agent = Agent(
            model=model,
            instructions=definition.system_prompt,
            output_type=str,
        )
        register_agent_tools(
            agent, 
            enable_retrieval=definition.enable_retrieval
        )
        return agent

    async def run(
        self, definition: AgentDefinition, 
        messages: list[ChatMessage], 
        temperature: float | None, 
        max_tokens: int | None
    ) -> tuple[str, ChatCompletionUsage]:
        agent = self.build_agent(definition)
        latest_user_prompt, message_history = self._split_messages(messages, definition.system_prompt)
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

    async def stream(
        self,
        definition: AgentDefinition,
        messages: list[ChatMessage],
        temperature: float | None,
        max_tokens: int | None,
    ):
        agent = self.build_agent(definition)
        latest_user_prompt, message_history = self._split_messages(messages, definition.system_prompt)
        return agent.run_stream(
            latest_user_prompt,
            deps=AgentDeps(retriever=self.retriever, messages=messages),
            message_history=message_history,
            model_settings=ModelSettings(
                temperature=temperature if temperature is not None else self.settings.llm_temperature,
                max_tokens=max_tokens if max_tokens is not None else self.settings.llm_max_tokens,
            ),
            usage_limits=self._build_usage_limits(),
        )

    def _split_messages(
        self,
        messages: list[ChatMessage],
        instructions: str,
    ) -> tuple[str, list[ModelMessage]]:
        latest_user_index = None
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].role == "user" and messages[index].content:
                latest_user_index = index
                break
        if latest_user_index is None:
            raise ValueError("user message content is required")

        latest_user_prompt = messages[latest_user_index].content or ""
        history = messages[:latest_user_index]
        extra_instructions = "\n\n".join(
            message.content
            for message in history
            if message.role == "system" and message.content
        )
        merged_instructions = self._merge_instructions(instructions, extra_instructions)
        model_messages: list[ModelMessage] = []
        first_request = True

        for message in history:
            if not message.content:
                continue
            if message.role == "system":
                continue
            if message.role == "assistant":
                model_messages.append(ModelResponse(parts=[TextPart(content=message.content)]))
                continue
            if message.role == "user":
                request_instructions = None
                if first_request:
                    request_instructions = merged_instructions
                model_messages.append(
                    ModelRequest.user_text_prompt(
                        message.content,
                        instructions=request_instructions,
                    )
                )
                first_request = False

        if not model_messages:
            if merged_instructions != instructions:
                return latest_user_prompt, [ModelRequest(parts=[], instructions=merged_instructions)]
            return latest_user_prompt, []

        first_message = model_messages[0]
        if isinstance(first_message, ModelRequest):
            if not first_message.instructions:
                first_message.instructions = merged_instructions
        else:
            model_messages.insert(0, ModelRequest(parts=[], instructions=merged_instructions))
        return latest_user_prompt, model_messages

    def _merge_instructions(self, base_instructions: str, extra_instruction: str) -> str:
        extra_instruction = extra_instruction.strip()
        if not extra_instruction:
            return base_instructions
        return f"{base_instructions}\n\n{extra_instruction}"

    def _build_usage_limits(self) -> UsageLimits:
        return UsageLimits(request_limit=10)
