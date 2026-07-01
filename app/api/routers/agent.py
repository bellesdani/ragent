from app.core.agent.service import AgentService
from fastapi.responses import StreamingResponse
from app.api.streaming.agent import stream_chat_completion
from fastapi import APIRouter, Depends, HTTPException, Request
from app.api.schemas.agent import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelCard,
    ModelListResponse,
)


router = APIRouter(tags=["Agent - Chat (OpenAI compat)"])


def get_agent_service(request: Request) -> AgentService:
    return request.app.state.agent_service


@router.get(
    path="/v1/models", 
    response_model=ModelListResponse,
    summary="Lista los modelos",
    description="Lista los modelos/agentes públicos disponibles en formato OpenAI",
)
async def list_public_agents(agent_service: AgentService = Depends(get_agent_service)) -> ModelListResponse:
    return ModelListResponse(
        data=[
            ModelCard(
                id=agent.agent_id,
                name=agent.name,
                description=agent.description,
            )
            for agent in agent_service.list_public_agents()
        ]
    )


@router.post(
    path="/v1/chat/completions", 
    response_model=ChatCompletionResponse,
    summary="Inicia o continua una conversación con un modelo",
    description="Inicia o continua una conversación con un modelo/agente seleccionado siguiendo el esquema de OpenAI",
)
async def create_chat_completion(
    request: ChatCompletionRequest, 
    agent_service: AgentService = Depends(get_agent_service),
) -> ChatCompletionResponse | StreamingResponse:
    try:
        result = await agent_service.complete_chat(
            model=request.model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if request.stream:
        # Voy a simular un formato de respuesta en Streaming por temas de compatibilidad con N8N
        return StreamingResponse(
            stream_chat_completion(model=result.model, content=result.content, usage=result.usage),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return ChatCompletionResponse(
        model=result.model,
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionChoiceMessage(content=result.content),
            )
        ],
        usage=result.usage,
    )
