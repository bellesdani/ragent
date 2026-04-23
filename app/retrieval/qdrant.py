from __future__ import annotations

import re
import math
from typing import Any
from collections import Counter
from app.core.config import Settings
from qdrant_client import AsyncQdrantClient
from app.retrieval.base import BaseRetriever
from app.llm.openai_compat import OpenAICompatClient
from app.api.schemas.openai import ChatMessage, RetrievalDocument, RetrievedContext


WORD_PATTERN = re.compile(r"\w+", re.UNICODE)
DEFAULT_COLLECTION = "corporate_knowledge"
DEFAULT_VECTOR_NAME: str | None = None
DEFAULT_SEARCH_LIMIT = 12
DEFAULT_TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.2
DEFAULT_PAYLOAD_TEXT_KEYS = ("text", "content", "chunk", "page_content")
DEFAULT_ENABLE_KEYWORD_RERANK = True
DEFAULT_CONTEXT_MAX_CHARS = 12000


class QdrantRetriever(BaseRetriever):
    def __init__(self, settings: Settings, embedding_client: OpenAICompatClient) -> None:
        self.settings = settings
        self.embedding_client = embedding_client
        self.client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)

    async def retrieve(self, query: str, messages: list[ChatMessage]) -> RetrievedContext:
        rewritten_query = self._rewrite_query(query, messages)
        query_vector = await self.embedding_client.create_embedding(
            input_text=rewritten_query,
            model=self.settings.embedding_model,
        )
        search_kwargs: dict[str, Any] = {
            "collection_name": DEFAULT_COLLECTION,
            "query": query_vector,
            "limit": DEFAULT_SEARCH_LIMIT,
            "with_payload": True,
            "score_threshold": DEFAULT_SCORE_THRESHOLD,
        }
        if DEFAULT_VECTOR_NAME:
            search_kwargs["using"] = DEFAULT_VECTOR_NAME
        results = await self.client.query_points(**search_kwargs)
        points = results.points if hasattr(results, "points") else []
        documents = [self._point_to_document(point) for point in points]
        if DEFAULT_ENABLE_KEYWORD_RERANK:
            documents = self._keyword_rerank(rewritten_query, documents)
        documents = self._trim_context(documents[:DEFAULT_TOP_K])
        return RetrievedContext(query=rewritten_query, documents=documents)

    def _rewrite_query(self, query: str, messages: list[ChatMessage]) -> str:
        recent_turns = [message.content for message in messages[-4:] if message.content and message.role in {"user", "assistant"}]
        if len(recent_turns) <= 1:
            return query.strip()
        context = " ".join(recent_turns[:-1]).strip()
        if not context:
            return query.strip()
        return f"Contexto conversacional: {context}\nConsulta actual: {query.strip()}"

    def _point_to_document(self, point: Any) -> RetrievalDocument:
        payload = dict(point.payload or {})
        text = self._extract_text(payload)
        return RetrievalDocument(
            id=str(point.id),
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
