from app.core.config import get_settings
from app.agent.catalog import AgentCatalog
from fastapi import APIRouter, HTTPException
from app.agent.service import ChatAgentService
from fastapi.responses import StreamingResponse
from app.retrieval.qdrant import QdrantRetriever
from app.llm.openai_compat import OpenAICompatClient
from app.api.schemas.openai import ChatCompletionRequest, ChatCompletionResponse, ModelCard, ModelListResponse


router = APIRouter()


settings = get_settings()
agent_catalog = AgentCatalog(settings)
chat_client = OpenAICompatClient(
    base_url=settings.chat_base_url,
    api_key=settings.chat_api_key,
    timeout=settings.llm_timeout_seconds,
    provider=settings.chat_provider,
)
embedding_client = OpenAICompatClient(
    base_url=settings.embedding_base_url,
    api_key=settings.embedding_api_key,
    timeout=settings.llm_timeout_seconds,
    provider=settings.embedding_provider,
)
retriever = QdrantRetriever(settings=settings, embedding_client=embedding_client)
agent_service = ChatAgentService(
    settings=settings,
    retriever=retriever,
    llm_client=chat_client,
    agent_catalog=agent_catalog,
)


@router.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    return ModelListResponse(
        data=[
            ModelCard(
                id=agent.agent_id,
                owned_by=settings.app_name,
                description=agent.description or agent.name,
            )
            for agent in agent_catalog.list_agents()
        ]
    )


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        if request.stream:
            stream = agent_service.stream_chat_completion(request)
            return StreamingResponse(stream, media_type="text/event-stream")
        return await agent_service.create_chat_completion(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
