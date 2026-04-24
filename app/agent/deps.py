from __future__ import annotations

from dataclasses import dataclass
from app.retrieval.base import BaseRetriever
from app.api.schemas.openai import ChatMessage


@dataclass
class AgentDeps:
    retriever: BaseRetriever
    messages: list[ChatMessage]
