from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


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


class KnowledgeSourceSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("La query de búsqueda no puede estar vacía")
        return cleaned_value


class KnowledgeSourceSearchItem(BaseModel):
    id: str = Field()
    score: float = Field()
    content: str = Field()
    metadata: dict[str, Any] = Field()


class KnowledgeSourceSearchResult(BaseModel):
    query: str = Field()
    last_data_update: str | None = Field(default=None)
    items: list[KnowledgeSourceSearchItem]
    count: int = Field()


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


class KnowledgeSourceSearchResponse(BaseModel):
    status: Literal["ok"] = "ok"
    operation: Literal["search"] = "search"
    knowledge_source_id: str
    result: KnowledgeSourceSearchResult


class KnowledgeSourceError(BaseModel):
    code: str
    message: str
    details: Any | None = None


class KnowledgeSourceErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    error: KnowledgeSourceError
