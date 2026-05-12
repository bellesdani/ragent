from typing import Any, Literal
from pydantic import BaseModel, Field


class KnowledgeSourcePayloadKeys(BaseModel):
    metadata_key: str
    semantic_content_key: str
    lexical_content_key: str | None = None


class KnowledgeSourceItem(BaseModel):
    id: str
    name: str
    description: str
    collection_name: str
    payload_keys: KnowledgeSourcePayloadKeys
    dense_vector_name: str | None
    sparse_vector_name: str | None
    retrieval_type: Literal["semantic", "lexical", "hybrid"]


class KnowledgeSourcesListResult(BaseModel):
    items: list[KnowledgeSourceItem]
    count: int


class KnowledgeSourceCreateResult(BaseModel):
    collection_created: bool


class KnowledgeSourceUpsertResult(BaseModel):
    points: int = 0
    summary: dict[str, Any] = Field(default_factory=dict)


class KnowledgeSourcesListResponse(BaseModel):
    status: Literal["ok"] = "ok"
    operation: Literal["list"] = "list"
    result: KnowledgeSourcesListResult


class KnowledgeSourceCreateResponse(BaseModel):
    status: Literal["ok"] = "ok"
    operation: Literal["create"] = "create"
    knowledge_source_id: str
    result: KnowledgeSourceCreateResult


class KnowledgeSourceUpsertResponse(BaseModel):
    status: Literal["ok"] = "ok"
    operation: Literal["upsert"] = "upsert"
    knowledge_source_id: str
    result: KnowledgeSourceUpsertResult


class KnowledgeSourceError(BaseModel):
    code: str
    message: str
    details: Any | None = None


class KnowledgeSourceErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    error: KnowledgeSourceError
