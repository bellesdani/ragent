from typing import List
from fastapi import APIRouter, Depends, Request
from app.core.entities import KnowledgeSource, TicketArticleRow
from app.core.knowledge_source.service import KnowledgeSourceService


router = APIRouter(tags=["Knowledge Sources"])


def get_knowledge_service(request: Request) -> KnowledgeSourceService:
    return request.app.state.knowledge_service


@router.get("/knowledge_source")
async def get_knowledge_sources(
    knowledge_service: KnowledgeSourceService = Depends(get_knowledge_service)
) -> List[KnowledgeSource]:
    
    return knowledge_service.list_knowledge_sources()


@router.post("/knowledge_source/tickets")
async def create_knowledge_source(
    knowledge_service: KnowledgeSourceService = Depends(get_knowledge_service)
) -> dict[str, object]:
    
    collection_created = await knowledge_service.create_knowledge_source("tickets")
    return {
        "status": "ok",
        "collection_created": collection_created,
    }


@router.post("/knowledge_source/tickets/points")
async def upsert_knowledge_source_data(
    ticketArticles: list[TicketArticleRow],
    knowledge_service: KnowledgeSourceService = Depends(get_knowledge_service),
) -> dict[str, object]:
    
    result = await knowledge_service.upsert_knowledge_source_data("tickets", ticketArticles)
    return {
        "status": "ok",
        "tickets": result["tickets"],
        "points": result["points"],
    }
