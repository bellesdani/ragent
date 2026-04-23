from __future__ import annotations

import re
import math
import logging
from typing import Any
from collections import Counter
from dataclasses import dataclass
from app.config.config import Settings
from qdrant_client import AsyncQdrantClient
from app.retrieval.base import BaseRetriever
from app.llm.openai_compat import OpenAICompatClient
from app.api.schemas.openai import ChatMessage, RetrievalDocument, RetrievedContext


logger = logging.getLogger(__name__)


DEFAULT_TOP_K = 5
DEFAULT_SEARCH_LIMIT = 12
DEFAULT_SCORE_THRESHOLD = 0.2
DEFAULT_CONTEXT_MAX_CHARS = 12000
DEFAULT_ENABLE_KEYWORD_RERANK = True
WORD_PATTERN = re.compile(r"\w+", re.UNICODE)
DEFAULT_PAYLOAD_TEXT_KEYS = ("text", "content", "chunk", "page_content")
DEFAULT_SEARCH_SOURCES = (
    {
        "id": "devices",
        "name": "Devices",
        "description": "Información sobre los dispostivos de la empresa, tales como servidores, equipos de usuarios, etc.",
        "collection": "devices",
        "vector_name": None,
    },
    {
        "id": "employees",
        "name": "Employees",
        "description": "Información sobre los empreados de la empresa y su contacto corporativo, como correo electónico, teléfono y extensión.",
        "collection": "employees",
        "vector_name": None,
    },
)


@dataclass(frozen=True)
class KnowledgeSource:
    id: str
    name: str
    description: str
    collection: str
    vector_name: str | None = None


class QdrantRetriever(BaseRetriever):
    def __init__(self, settings: Settings, embedding_client: OpenAICompatClient) -> None:
        self.settings = settings
        self.embedding_client = embedding_client
        self.client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
        self.sources = {
            source["id"]: KnowledgeSource(**source)
            for source in DEFAULT_SEARCH_SOURCES
        }

    async def retrieve(self, query: str, messages: list[ChatMessage]) -> RetrievedContext:
        return await self.retrieve_from_sources(query=query, messages=messages, source_ids=None)

    async def retrieve_from_sources(self, query: str, messages: list[ChatMessage], source_ids: list[str] | None = None) -> RetrievedContext:
        rewritten_query = self._rewrite_query(query, messages)
        logger.debug(
            "Starting retrieval from sources | query=%s rewritten_query=%s source_ids=%s",
            query,
            rewritten_query,
            source_ids,
        )
        query_vector = await self.embedding_client.create_embedding(
            input_text=rewritten_query,
            model=self.settings.embedding_model,
        )
        selected_sources = self._resolve_sources(source_ids)
        documents: list[RetrievalDocument] = []
        for source in selected_sources:
            logger.debug(
                "Querying Qdrant source | source_id=%s collection=%s vector_name=%s",
                source.id,
                source.collection,
                source.vector_name,
            )
            search_kwargs: dict[str, Any] = {
                "collection_name": source.collection,
                "query": query_vector,
                "limit": DEFAULT_SEARCH_LIMIT,
                "with_payload": True,
                "score_threshold": DEFAULT_SCORE_THRESHOLD,
            }
            if source.vector_name:
                search_kwargs["using"] = source.vector_name
            results = await self.client.query_points(**search_kwargs)
            points = results.points if hasattr(results, "points") else []
            logger.debug("Qdrant source response | source_id=%s points_found=%d", source.id, len(points))
            documents.extend(self._point_to_document(point, source) for point in points)
        if DEFAULT_ENABLE_KEYWORD_RERANK:
            documents = self._keyword_rerank(rewritten_query, documents)
        documents = self._trim_context(documents[:DEFAULT_TOP_K])
        logger.debug(
            "Retrieval completed | documents=%d documents_preview=%s",
            len(documents),
            [
                {
                    "id": document.id,
                    "source": document.metadata.get("source_name") or document.metadata.get("collection"),
                    "score": document.score,
                    "text_preview": document.text[:200],
                }
                for document in documents[:3]
            ],
        )
        return RetrievedContext(query=rewritten_query, documents=documents)

    def list_sources(self) -> list[dict[str, str]]:
        return [
            {
                "id": source.id,
                "name": source.name,
                "description": source.description,
            }
            for source in self.sources.values()
        ]

    def _rewrite_query(self, query: str, messages: list[ChatMessage]) -> str:
        recent_turns = [
            message.content
            for message in messages[-4:]
            if message.content and message.role in {"user", "assistant"}
        ]
        if len(recent_turns) <= 1:
            return query.strip()
        context = " ".join(recent_turns[:-1]).strip()
        if not context:
            return query.strip()
        return f"Contexto conversacional: {context}\nConsulta actual: {query.strip()}"

    def _point_to_document(self, point: Any, source: KnowledgeSource) -> RetrievalDocument:
        payload = dict(point.payload or {})
        text = self._extract_text(payload)
        payload.setdefault("collection", source.collection)
        payload.setdefault("source_id", source.id)
        payload.setdefault("source_name", source.name)
        return RetrievalDocument(
            id=f"{source.id}:{point.id}",
            score=float(point.score or 0.0),
            text=text,
            metadata=payload,
        )

    def _extract_text(self, payload: dict[str, Any]) -> str:
        for key in DEFAULT_PAYLOAD_TEXT_KEYS:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _keyword_rerank(self, query: str, documents: list[RetrievalDocument]) -> list[RetrievalDocument]:
        query_terms = self._term_counts(query)
        if not query_terms:
            return documents
        scored = []
        for doc in documents:
            doc_terms = self._term_counts(doc.text)
            lexical_score = self._cosine_similarity(query_terms, doc_terms)
            combined_score = (doc.score * 0.8) + (lexical_score * 0.2)
            scored.append((combined_score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored]

    def _trim_context(self, documents: list[RetrievalDocument]) -> list[RetrievalDocument]:
        current_size = 0
        trimmed: list[RetrievalDocument] = []
        for doc in documents:
            if not doc.text:
                continue
            next_size = current_size + len(doc.text)
            if trimmed and next_size > DEFAULT_CONTEXT_MAX_CHARS:
                break
            trimmed.append(doc)
            current_size = next_size
        return trimmed

    def _term_counts(self, text: str) -> Counter[str]:
        return Counter(token.lower() for token in WORD_PATTERN.findall(text))

    def _cosine_similarity(self, left: Counter[str], right: Counter[str]) -> float:
        intersection = set(left) & set(right)
        numerator = sum(left[term] * right[term] for term in intersection)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if not left_norm or not right_norm:
            return 0.0
        return numerator / (left_norm * right_norm)

    def _resolve_sources(self, source_ids: list[str] | None) -> list[KnowledgeSource]:
        if not source_ids:
            return list(self.sources.values())
        unknown = [source_id for source_id in source_ids if source_id not in self.sources]
        if unknown:
            available = ", ".join(sorted(self.sources))
            raise ValueError(f"Unknown knowledge sources: {', '.join(unknown)}. Available: {available}")
        return [self.sources[source_id] for source_id in source_ids]
