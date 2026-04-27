from __future__ import annotations


from fastapi import APIRouter, HTTPException
from app.core.chat import ChatAgentService
from app.core.config import Settings
from app.api.schemas import (
    ChatCompletionChoiceMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ModelListResponse,
    ModelCard,
)


def create_router(settings: Settings, chat_service: ChatAgentService) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/v1/models", response_model=ModelListResponse)
    async def list_models() -> ModelListResponse:
        return ModelListResponse(
            data=[
                ModelCard(
                    id=agent.agent_id,
                    name=agent.name,
                    description=agent.description,
                )
                for agent in chat_service.list_agents()
            ]
        )

    @router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
        if request.stream:
            # TODO: Not implemented yet
            pass

        try:
            result = await chat_service.complete(
                model=request.model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return ChatCompletionResponse(
            model=result.model,
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionChoiceMessage(content=result.content),
                )
            ],
            usage=result.usage,
        )

    return router
