from app.config import Settings
from pydantic_ai.usage import UsageLimits
from pydantic_ai.settings import ModelSettings
from app.core.agent.catalog import AgentCatalog
from app.core.agent.factory import AgentFactory
from pydantic_ai import UnexpectedModelBehavior
from app.core.utils.prompts import PromptService


class SummarizerService:
    """
    Servicio interno para generar resumenes auxiliares sin acoplar la ingesta de conocimiento al servicio publico de agentes.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.factory = AgentFactory(
            settings=settings,
        )
        self.agent_catalog = AgentCatalog(
            settings=settings,
            prompt_service=PromptService(),
        )


    async def summarize(self, content: str, temperature: float | None = None, max_tokens: int | None = None) -> str:
        agent_definition = self.agent_catalog.get_agent("Summarizer")
        agent = self.factory.build(agent_definition)

        try:
            result = await agent.run(
                content,
                usage_limits=UsageLimits(request_limit=1),
                model_settings=ModelSettings(
                    temperature=temperature if temperature is not None else self.settings.llm_temperature,
                    max_tokens=max_tokens if max_tokens is not None else self.settings.llm_max_tokens,
                ),
            )
            return result.output
        except UnexpectedModelBehavior:
            return content
