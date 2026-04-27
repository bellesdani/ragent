from __future__ import annotations

from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import TYPE_CHECKING, Any, Literal


if TYPE_CHECKING:
    from app.core.retrieval import QdrantRetriever


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class RetrievalDocument(BaseModel):
    id: str
    score: float
    text: str
    metadata: dict[str, Any]


class RetrievedContext(BaseModel):
    query: str
    documents: list[RetrievalDocument]


@dataclass(frozen=True)
class ChatResult:
    model: str
    content: str
    usage: ChatCompletionUsage


@dataclass(frozen=True)
class AgentDefinition:
    agent_id: str
    name: str
    description: str
    backend_base_url: str
    backend_api_key: str
    backend_chat_model: str
    system_prompt: str
    enable_retrieval: bool


@dataclass
class AgentDeps:
    retriever: QdrantRetriever
    messages: list[ChatMessage]


@dataclass(frozen=True)
class KnowledgeSource:
    id: str
    name: str
    description: str
    collection: str
    vector_name: str | None = None
