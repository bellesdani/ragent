from typing import List
from fastapi import APIRouter, Depends, Request
from app.core.entities import KnowledgeSourceDefinition
from app.core.knowledge_source.service import KnowledgeSourceService


router = APIRouter(tags=["Knowledge Sources"])


def get_knowledge_service(request: Request) -> KnowledgeSourceService:
    return request.app.state.knowledge_service


@router.get("/knowledge-source")
async def get_knowledge_sources(
    knowledge_service: KnowledgeSourceService = Depends(get_knowledge_service)
) -> List[KnowledgeSourceDefinition]:
    return knowledge_service.list_knowledge_sources()


@router.post("/knowledge-source/{knowledge_source_id}")
async def create_knowledge_source(
    knowledge_source_id: str,
    knowledge_service: KnowledgeSourceService = Depends(get_knowledge_service)
) -> dict[str, object]:
    collection_created = await knowledge_service.create_knowledge_source(knowledge_source_id)
    return {
        "status": "ok",
        "collection_created": collection_created,
    }


@router.post("/knowledge-source/{knowledge_source_id}/points")
async def upsert_knowledge_source_data(
    knowledge_source_id: str,
    data: list[dict[str, object]],
    knowledge_service: KnowledgeSourceService = Depends(get_knowledge_service),
) -> dict[str, object]:
    result = await knowledge_service.upsert_knowledge_source_data(knowledge_source_id, data)
    return {
        "status": "ok",
        **result,
    }
