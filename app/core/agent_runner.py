from __future__ import annotations


from app.core.config import Settings
from pydantic_ai.usage import UsageLimits
from pydantic_ai.settings import ModelSettings
from app.core.agent_factory import AgentFactory
from app.core.qdrant_retrieval import QdrantRetriever
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart
from app.core.entities import AgentDefinition, AgentDeps, ChatCompletionUsage, ChatMessage


class AgentRunner:

    def __init__(self, settings: Settings, retriever: QdrantRetriever) -> None:
        self.settings = settings
        self.retriever = retriever
        self.factory = AgentFactory(settings)


    async def run(self, definition: AgentDefinition, messages: list[ChatMessage], temperature: float | None, max_tokens: int | None) -> tuple[str, ChatCompletionUsage]:
        # Construimos nuestro agente
        agent = self.factory.build(definition)
        # Cargamos el historial de mensajes
        latest_user_prompt, message_history = split_messages(messages, definition.system_prompt)
        # Llamamos al run de PydanticAI para ejecutar el agente con toda la configuración establecida 
        try:
            result = await agent.run(
                latest_user_prompt,
                deps=AgentDeps(retriever=self.retriever, messages=messages),
                message_history=message_history,
                model_settings=ModelSettings(
                    temperature=temperature if temperature is not None else self.settings.llm_temperature,
                    max_tokens=max_tokens if max_tokens is not None else self.settings.llm_max_tokens,
                ),
                usage_limits=UsageLimits(request_limit=10),
            )
        except UnexpectedModelBehavior:
            # Esto puede ocurrir si el modelo intenta utilizar una herramienta o la consulta no puede ser resuelta,  
            #  y el modelo termina devolviendo un resultado vacío
            return (
                "Lo siento, no he sabido responder a esa pregunta.", 
                ChatCompletionUsage(
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                )
            )
        # Si no hay nada raro, devolvemos la respuesta del agente y las métricas de uso
        usage = result.usage()
        return result.output, ChatCompletionUsage(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        )


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

