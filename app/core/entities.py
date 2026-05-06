import time
import uuid

from datetime import datetime
from pydantic import BaseModel
from dataclasses import dataclass
from typing import Literal, Optional
from typing import Any, Literal, TYPE_CHECKING
from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from app.core.knowledge_source.retrieval import QdrantRetriever


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
    enable_tools: bool
    public: bool


@dataclass
class AgentDeps:
    retriever: "QdrantRetriever"
    messages: list[ChatMessage]


@dataclass(frozen=True)
class KnowledgeSource:
    id: str
    name: str
    description: str
    collection: str
    vector_name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    stream: bool = False
    user: str | None = None


class ChatCompletionChoiceMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionChoiceMessage
    finish_reason: Literal["stop", "length"] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ModelCard(BaseModel):
    id: str
    name: str
    description: str


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class TicketArticleRow(BaseModel):
    ticket_id: int
    ticket_group_id: int
    ticket_priority_id: int
    ticket_state_id: int
    ticket_organization_id: Optional[int] = None
    ticket_number: str
    ticket_title: str
    ticket_created_at: datetime
    ticket_closed_at: datetime

    ticket_customer_firstname: Optional[str] = None
    ticket_customer_lastname: Optional[str] = None
    ticket_customer_department: Optional[str] = None
    ticket_customer_email: Optional[str] = None

    ticket_creator_firstname: Optional[str] = None
    ticket_creator_lastname: Optional[str] = None
    ticket_creator_department: Optional[str] = None
    ticket_creator_email: Optional[str] = None

    article_id: int
    article_from: Optional[str] = None
    article_to: Optional[str] = None
    article_subject: Optional[str] = None
    article_content_type: Optional[str] = None
    article_body: Optional[str] = None
    article_internal: bool
    article_created_at: datetime

    article_creator_firstname: Optional[str] = None
    article_creator_lastname: Optional[str] = None
    article_creator_department: Optional[str] = None
    article_creator_email: Optional[str] = None


class TicketArticle(BaseModel):
    article_id: int
    article_from: Optional[str] = None
    article_to: Optional[str] = None
    article_subject: Optional[str] = None
    article_content_type: Optional[str] = None
    article_body: Optional[str] = None
    article_internal: bool
    article_created_at: datetime

    article_creator_firstname: Optional[str] = None
    article_creator_lastname: Optional[str] = None
    article_creator_department: Optional[str] = None
    article_creator_email: Optional[str] = None


class Ticket(BaseModel):
    ticket_id: int
    ticket_group_id: int
    ticket_priority_id: int
    ticket_state_id: int
    ticket_organization_id: Optional[int] = None
    ticket_number: str
    ticket_title: str
    ticket_created_at: datetime
    ticket_closed_at: datetime

    ticket_customer_firstname: Optional[str] = None
    ticket_customer_lastname: Optional[str] = None
    ticket_customer_department: Optional[str] = None
    ticket_customer_email: Optional[str] = None

    ticket_creator_firstname: Optional[str] = None
    ticket_creator_lastname: Optional[str] = None
    ticket_creator_department: Optional[str] = None
    ticket_creator_email: Optional[str] = None

    articles: list[TicketArticle]
