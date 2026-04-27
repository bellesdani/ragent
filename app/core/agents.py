from __future__ import annotations

from openai import AsyncOpenAI
from pydantic_ai.usage import UsageLimits
from pydantic_ai.settings import ModelSettings
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart

from app.core.config import Settings
from app.core.prompts import load_prompt
from app.core.retrieval import QdrantRetriever
from app.core.models import AgentDefinition, AgentDeps, ChatCompletionUsage, ChatMessage


class AgentCatalog:
    def __init__(self, settings: Settings) -> None:
        self._agents = {
            "Quipi": AgentDefinition(
                agent_id="Quipi",
                name="Quipi",
                description="Agente conversacional corporativo con acceso a herramientas de busqueda sobre Qdrant.",
                backend_base_url=settings.chat_base_url,
                backend_api_key=settings.chat_api_key,
                backend_chat_model=settings.chat_model,
                system_prompt=load_prompt("agents/quipi_system.md"),
                enable_retrieval=True,
            ),
            "Base": AgentDefinition(
                agent_id="Base",
                name="Base",
                description="Agente conversacional basico sin conexiones a herramientas de busqueda.",
                backend_base_url=settings.chat_base_url,
                backend_api_key=settings.chat_api_key,
                backend_chat_model=settings.chat_model,
                system_prompt=load_prompt("agents/base_system.md"),
                enable_retrieval=False,
            ),
        }

    def list_agents(self) -> list[AgentDefinition]:
        return list(self._agents.values())

    def get_agent(self, agent_id: str) -> AgentDefinition:
        if agent_id in self._agents:
            return self._agents[agent_id]
        raise ValueError(f"Unknown agent '{agent_id}'. Available agents: {', '.join(sorted(self._agents))}")


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


def split_messages(messages: list[ChatMessage], instructions: str) -> tuple[str, list[ModelMessage]]:
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
    merged_instructions = merge_instructions(instructions, extra_instructions)
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


def merge_instructions(base_instructions: str, extra_instruction: str) -> str:
    extra_instruction = extra_instruction.strip()
    if not extra_instruction:
        return base_instructions
    return f"{base_instructions}\n\n{extra_instruction}"


def register_retrieval_tool(agent: Agent[AgentDeps, str]) -> None:
    @agent.tool
    async def search_sources(ctx: RunContext[AgentDeps], query: str) -> str:
        """
        Busca informacion en Qdrant usando una o varias colecciones.

        Args:
            query: Consulta a buscar en la base de conocimiento.
            source_ids: Lista opcional de colecciones a consultar. Si se omite, busca en todas.
        """
        retrieval = await ctx.deps.retriever.retrieve(
            query=query,
            messages=ctx.deps.messages,
        )
        if not retrieval.documents:
            return "No se encontro evidencia relevante en las fuentes solicitadas."

        lines = [f"Consulta reescrita: {retrieval.query}"]
        for index, document in enumerate(retrieval.documents, start=1):
            source = (
                document.metadata.get("source_name")
                or document.metadata.get("collection")
                or document.id
            )
            lines.append(f"[{index}] Fuente: {source}")
            lines.append(f"[{index}] Score: {document.score:.4f}")
            lines.append(f"[{index}] Texto: {document.text}")
        return "\n".join(lines)


def register_calculator_tool(agent: Agent[AgentDeps, str]) -> None:
    @agent.tool_plain
    def calculator(expression: str) -> str:
        """
        Evalua una expresion aritmetica simple.

        Args:
            expression: Expresion matematica usando numeros, parentesis y operadores basicos.
        """
        if not expression.strip():
            raise ModelRetry("La expresion no puede estar vacia.")
        if any(char not in "0123456789+-*/()., %\t\n" for char in expression):
            raise ModelRetry("La expresion contiene caracteres no permitidos.")
        normalized = expression.replace(",", ".")
        try:
            result = eval(normalized, {"__builtins__": {}}, {})
        except Exception as exc:  # pragma: no cover - depende de la expresion del usuario
            raise ModelRetry(f"No se pudo evaluar la expresion: {exc}") from exc
        return str(result)
