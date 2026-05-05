from typing import List
from fastapi import APIRouter
from app.config import get_settings
from app.core.embeddings import EmbeddingClient
from app.core.qdrant.ingestion import QdrantIngestor
from app.core.entities import KnowledgeSource, TicketArticleRow
from app.core.qdrant.knowledge_sources import QdrantKnowledgeSourceCatalog

router = APIRouter(tags=["Knowledge Sources"])


@router.get("/knowledge_sources")
async def get_knowledge_sources() -> List[KnowledgeSource]:
    catalog = QdrantKnowledgeSourceCatalog()
    return catalog.get_knowledge_sources()


@router.post("/knowledge_sources/tickets")
async def create_tickets_collection() -> dict[str, str]:
    settings = get_settings()
    embedding_client = EmbeddingClient(
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
        timeout=settings.llm_timeout_seconds
    )
    qdrant_ingestor = QdrantIngestor(settings=settings, embedding_client=embedding_client)
    result = qdrant_ingestor.create_tickets_collection()
    return {
        "status": "ok"
    }


@router.post("/knowledge_sources/tickets/points")
async def create_tickets_points(ticketArticles: list[TicketArticleRow]) -> dict[str, str]:
    settings = get_settings()
    embedding_client = EmbeddingClient(
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
        timeout=settings.llm_timeout_seconds
    )
    qdrant_ingestor = QdrantIngestor(settings=settings, embedding_client=embedding_client)
    result = qdrant_ingestor.upsert_tickets_points(ticketArticles)
    return {
        "status": "ok"
    }
