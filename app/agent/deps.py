from __future__ import annotations

from dataclasses import dataclass

from app.api.schemas.openai import ChatMessage
from app.retrieval.base import BaseRetriever


@dataclass
class AgentDeps:
    retriever: BaseRetriever
    messages: list[ChatMessage]
