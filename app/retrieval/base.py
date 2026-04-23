from __future__ import annotations
from abc import ABC, abstractmethod
from app.api.schemas.openai import ChatMessage, RetrievedContext


class BaseRetriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, messages: list[ChatMessage]) -> RetrievedContext:
        raise NotImplementedError

    @abstractmethod
    async def retrieve_from_sources(self, query: str, messages: list[ChatMessage], source_ids: list[str] | None = None) -> RetrievedContext:
        raise NotImplementedError

    @abstractmethod
    def list_sources(self) -> list[dict[str, str]]:
        raise NotImplementedError
