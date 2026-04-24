from __future__ import annotations

from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from app.agent.definitions import AgentDefinition
from app.agent.deps import AgentDeps
from app.agent.tools import register_calculator_tool, register_retrieval_tool
from app.config.config import Settings


class AgentFactory:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def build(self, definition: AgentDefinition) -> Agent[AgentDeps, str]:
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
        if definition.enable_retrieval:
            register_retrieval_tool(agent)
        register_calculator_tool(agent)
        return agent
