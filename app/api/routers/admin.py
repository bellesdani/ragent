from fastapi import APIRouter, Depends, Request
from app.config import Settings
from app.core.agent.service import AgentService
from app.core.knowledge_source.service import KnowledgeSourceService
from app.core.agent.factory import RETRIEVAL_TOOL_NAMES, UTILITY_TOOL_NAMES
from app.api.schemas.admin import (
    AdminAgentItem,
    AdminAgentTools,
    AdminChatBackend,
    AdminQdrantInfo,
    AdminServiceInfo,
    AdminSourceStatus,
    AdminAgentsSummary,
    AdminAgentsResponse,
    AdminEmbeddingBackend,
    AdminOverviewResponse,
)


ADMIN_TAG = "Admin - Panel"


router = APIRouter()


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_agent_service(request: Request) -> AgentService:
    return request.app.state.agent_service


def get_knowledge_service(request: Request) -> KnowledgeSourceService:
    return request.app.state.knowledge_service


@router.get(
    path="/admin/overview",
    response_model=AdminOverviewResponse,
    tags=[ADMIN_TAG],
    summary="Estado general del servicio",
    description="Devuelve el estado del servicio, del backend de chat, de Qdrant y de las fuentes de conocimiento configuradas. Da soporte al panel de administración.",
)
async def get_admin_overview(
    request: Request,
    settings: Settings = Depends(get_settings),
    agent_service: AgentService = Depends(get_agent_service),
    knowledge_service: KnowledgeSourceService = Depends(get_knowledge_service),
) -> AdminOverviewResponse:
    # Comprobamos la conexión con Qdrant y contamos las colecciones existentes
    qdrant_status = "ok"
    collections_total = 0
    try:
        collections = await knowledge_service.qdrant_client.get_collections()
        collections_total = len(collections.collections)
    except Exception:
        qdrant_status = "unreachable"

    # Para cada fuente configurada, consultamos su colección (puntos y última actualización)
    sources: list[AdminSourceStatus] = []
    for definition in knowledge_service.list_knowledge_sources():
        collection_exists = False
        points = None
        last_update = None
        if qdrant_status == "ok":
            try:
                collection_info = await knowledge_service.qdrant_client.get_collection(
                    collection_name=definition.collection_name,
                )
                collection_exists = True
                points = collection_info.points_count
                if collection_info.config.metadata:
                    last_update = collection_info.config.metadata.get("last_collection_update")
            except Exception:
                collection_exists = False
        sources.append(
            AdminSourceStatus(
                id=definition.id,
                name=definition.name,
                description=definition.description,
                collection_name=definition.collection_name,
                retrieval_type=definition.retrieval_type,
                collection_exists=collection_exists,
                points=points,
                last_update=last_update,
            )
        )

    agents = agent_service.list_agents()
    return AdminOverviewResponse(
        service=AdminServiceInfo(
            name=request.app.title,
            version=request.app.version,
        ),
        chat_backend=AdminChatBackend(
            model=settings.chat_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout_seconds=settings.llm_timeout_seconds,
        ),
        embedding=AdminEmbeddingBackend(
            model=settings.embedding_model,
        ),
        qdrant=AdminQdrantInfo(
            url=settings.qdrant_url,
            status=qdrant_status,
            collections_total=collections_total,
        ),
        agents=AdminAgentsSummary(
            total=len(agents),
            public=sum(1 for agent in agents if agent.public),
            names=[agent.name for agent in agents],
        ),
        sources=sources,
    )


@router.get(
    path="/admin/agents",
    response_model=AdminAgentsResponse,
    tags=[ADMIN_TAG],
    summary="Lista todos los agentes definidos",
    description="Lista todas las definiciones de agentes del catálogo (públicos e internos), con su prompt y sus herramientas. Da soporte al panel de administración.",
)
async def list_admin_agents(
    agent_service: AgentService = Depends(get_agent_service),
) -> AdminAgentsResponse:
    items = [
        AdminAgentItem(
            agent_id=agent.agent_id,
            name=agent.name,
            description=agent.description,
            model=agent.backend_chat_model,
            public=agent.public,
            enable_tools=agent.enable_tools,
            system_prompt=agent.system_prompt,
            system_prompt_file=agent.system_prompt_file,
            tools=AdminAgentTools(
                retrieval=list(RETRIEVAL_TOOL_NAMES) if agent.enable_tools else [],
                utility=list(UTILITY_TOOL_NAMES) if agent.enable_tools else [],
            ),
        )
        for agent in agent_service.list_agents()
    ]
    return AdminAgentsResponse(
        items=items,
        count=len(items),
    )
