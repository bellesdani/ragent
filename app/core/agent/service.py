from app.config import Settings
from pydantic_ai.usage import UsageLimits
from pydantic_ai.settings import ModelSettings
from pydantic_ai import UnexpectedModelBehavior
from app.core.agent.factory import AgentFactory
from app.core.embeddings import EmbeddingClient
from app.core.agent.catalog import AgentCatalog
from app.core.knowledge_source.retrieval import QdrantRetriever
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart
from app.core.entities import AgentDefinition, AgentDeps, ChatCompletionUsage, ChatMessage, ChatResult


class AgentService:
    """
    Este servicio es el punto de acceso y gestor de agentes. Utiliza:
     - Las variables cargadas (Settings)
     - El catálogo de agentes (AgentCatalog)
     - La factoría de agentes para construirlos (AgentFactory)
     - El servicio de búsqueda de Qdrant (QdrantRetriever)
    
    Funciones públicas:
     - Listar los agentes disponibles (list_agents).
     - Realizar un chat completion con un agente concreto (complete_chat).
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.agent_catalog = AgentCatalog(settings)
        self.factory = AgentFactory(settings)
        self.retriever = QdrantRetriever(
            settings=settings, 
            embedding_client=EmbeddingClient(
                base_url=settings.embedding_base_url,
                api_key=settings.embedding_api_key,
                timeout=settings.llm_timeout_seconds,
            )
        )


    def list_agents(self) -> list[AgentDefinition]:
        return self.agent_catalog.list_agents()


    async def complete_chat(self, model: str, messages: list[ChatMessage], temperature: float | None, max_tokens: int | None) -> ChatResult:
        # Construimos el agente
        agent_definition = self.agent_catalog.get_agent(model)
        agent = self.factory.build(agent_definition)

        # Cargamos el historial de mensajes
        latest_user_prompt, message_history = self._split_messages(messages, agent_definition.system_prompt)

        # Llamamos al run de PydanticAI para ejecutar el agente con toda la configuración establecida 
        try:
            result = await agent.run(
                latest_user_prompt,
                message_history=message_history,
                usage_limits=UsageLimits(request_limit=10),
                deps=AgentDeps(
                    retriever=self.retriever, 
                    messages=messages
                ),
                model_settings= ModelSettings(
                    temperature=temperature if temperature is not None else self.settings.llm_temperature,
                    max_tokens=max_tokens if max_tokens is not None else self.settings.llm_max_tokens,
                ),
            )
            # Si no hay nada raro, devolvemos la respuesta del agente y las métricas de uso
            usage = result.usage()
            return ChatResult(
                model=agent_definition.agent_id,
                content=result.output,
                usage=ChatCompletionUsage(
                    prompt_tokens=usage.input_tokens,
                    completion_tokens=usage.output_tokens,
                    total_tokens=usage.total_tokens,
                ),
            )

        except UnexpectedModelBehavior:
            # Esto puede ocurrir si el modelo intenta utilizar una herramienta o la consulta no puede ser resuelta,  
            #  y el modelo termina devolviendo un resultado vacío
            return ChatResult(
                model=agent_definition.agent_id,
                content="Lo siento, no he sabido responder a esa pregunta.",
                usage=ChatCompletionUsage(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                ),
            )

    
    def _split_messages(self, messages: list[ChatMessage], instructions: str) -> tuple[str, list[ModelMessage]]:
        latest_user_index = None
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].role == "user" and messages[index].content:
                latest_user_index = index
                break
        if latest_user_index is None:
            raise ValueError("Se necesita un mensaje del usuario")

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
