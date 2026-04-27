from __future__ import annotations

from dataclasses import dataclass
from app.api.schemas.openai import ChatMessage
from app.retrieval.qdrant import QdrantRetriever


@dataclass
class AgentDeps:
    retriever: QdrantRetriever
    messages: list[ChatMessage]
