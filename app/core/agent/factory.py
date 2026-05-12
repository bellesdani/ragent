from pydantic_ai import Agent
from openai import AsyncOpenAI
from app.config import Settings
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from app.core.agent.entities import AgentDefinition, AgentDeps
from app.core.agent.tools import (
    register_datetime_tool,
    register_calculator_tool, 
    register_devices_retrieval_tool,
    register_manuals_retrieval_tool,
    register_tickets_retrieval_tool,
    register_employees_retrieval_tool,
)


class AgentFactory:
    """
    Esta factoría construye agentes a partir de una definición. Utiliza:
     - Las variables cargadas (Settings) para definir los proveedores y modelos.

    Funciones públicas:
     - Construir un agente ejecutable (build).
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def build(self, definition: AgentDefinition) -> Agent[AgentDeps, str]:
        # Definimos el tipo de proveedor OpenAI (vLLM está desplegado siguiendo la API de OpenAI)
        provider = OpenAIProvider(
            openai_client=AsyncOpenAI(
                base_url=definition.backend_base_url.rstrip("/") + "/",
                api_key=definition.backend_api_key or "api-key-not-set",
                timeout=self.settings.llm_timeout_seconds,
            )
        )
        # Definimos el modelo y los hiperparámetros base del agente
        model = OpenAIChatModel(
            definition.backend_chat_model,
            provider=provider,
            settings=ModelSettings(
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_tokens,
            ),
        )
        # Definimos el agente, con el modelo indicado y el prompt
        agent: Agent[AgentDeps, str] = Agent(
            model=model,
            instructions=definition.system_prompt,
            output_type=str,
        )
        # Definimos las tools que el agente puede utilizar 
        if definition.enable_tools:
            register_employees_retrieval_tool(agent)
            register_devices_retrieval_tool(agent)
            register_manuals_retrieval_tool(agent)
            register_tickets_retrieval_tool(agent)
            register_calculator_tool(agent)
            register_datetime_tool(agent)

        return agent
