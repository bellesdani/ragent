from typing import Literal
from pydantic import BaseModel


class AdminServiceInfo(BaseModel):
    name: str
    version: str


class AdminChatBackend(BaseModel):
    model: str
    temperature: float
    max_tokens: int
    timeout_seconds: float


class AdminEmbeddingBackend(BaseModel):
    model: str


class AdminQdrantInfo(BaseModel):
    url: str
    status: Literal["ok", "unreachable"]
    collections_total: int


class AdminAgentsSummary(BaseModel):
    total: int
    public: int
    names: list[str]


class AdminSourceStatus(BaseModel):
    id: str
    name: str
    description: str
    collection_name: str
    retrieval_type: Literal["semantic", "lexical", "hybrid"]
    collection_exists: bool
    points: int | None = None
    last_update: str | None = None


class AdminOverviewResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: AdminServiceInfo
    chat_backend: AdminChatBackend
    embedding: AdminEmbeddingBackend
    qdrant: AdminQdrantInfo
    agents: AdminAgentsSummary
    sources: list[AdminSourceStatus]


class AdminAgentTools(BaseModel):
    retrieval: list[str]
    utility: list[str]


class AdminAgentItem(BaseModel):
    agent_id: str
    name: str
    description: str
    model: str
    public: bool
    enable_tools: bool
    system_prompt: str
    system_prompt_file: str | None = None
    tools: AdminAgentTools


class AdminAgentsResponse(BaseModel):
    status: Literal["ok"] = "ok"
    items: list[AdminAgentItem]
    count: int
