from __future__ import annotations
from abc import ABC, abstractmethod
from app.api.schemas.openai import ChatMessage, RetrievedContext


class BaseRetriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, messages: list[ChatMessage]) -> RetrievedContext:
        raise NotImplementedError
