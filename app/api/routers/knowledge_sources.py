from typing import List
from fastapi import APIRouter
from app.core.entities import KnowledgeSource
from app.core.qdrant_knowledge_sources import QdrantKnowledgeSourceCatalog


router = APIRouter(tags=["Knowledge Sources"])


@router.get("/knowledge_sources")
async def get_knowledge_sources() -> List[KnowledgeSource]:
    catalog = QdrantKnowledgeSourceCatalog()
    return catalog.get_knowledge_sources()

