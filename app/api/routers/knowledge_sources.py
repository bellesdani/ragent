from typing import List
from fastapi import APIRouter, Depends, Request
from app.core.qdrant.ingestion import QdrantIngestor
from app.core.entities import KnowledgeSource, TicketArticleRow
from app.core.qdrant.knowledge_sources import QdrantKnowledgeSourceCatalog

router = APIRouter(tags=["Knowledge Sources"])


def get_knowledge_service(request: Request) -> QdrantIngestor:
    return request.app.state.knowledge_service


@router.get("/knowledge_sources")
async def get_knowledge_sources() -> List[KnowledgeSource]:
    catalog = QdrantKnowledgeSourceCatalog()
    return catalog.get_knowledge_sources()


@router.post("/knowledge_sources/tickets")
async def create_tickets_collection(knowledge_service: QdrantIngestor = Depends(get_knowledge_service)) -> dict[str, object]:
    collection_created = await knowledge_service.create_tickets_collection()
    return {
        "status": "ok",
        "collection_created": collection_created,
    }


@router.post("/knowledge_sources/tickets/points")
async def create_tickets_points(
    ticketArticles: list[TicketArticleRow],
    knowledge_service: QdrantIngestor = Depends(get_knowledge_service),
) -> dict[str, object]:
    result = await knowledge_service.upsert_tickets_points(ticketArticles)
    return {
        "status": "ok",
        "tickets": result["tickets"],
        "points": result["points"],
    }
