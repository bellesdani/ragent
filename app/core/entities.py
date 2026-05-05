import time
import uuid

from typing import Literal
from pydantic import BaseModel
from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING
from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from app.core.qdrant_retrieval import QdrantRetriever


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

    @model_validator(mode="after")
    def validate_messages(self) -> "ChatCompletionRequest":
        if not self.messages:
            raise ValueError("messages must contain at least one message")
        if not any(message.role == "user" for message in self.messages):
            raise ValueError("messages must include at least one user message")
        return self


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



