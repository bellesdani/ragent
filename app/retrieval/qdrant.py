from __future__ import annotations

import logging

from typing import Any
from dataclasses import dataclass
from app.config.config import Settings
from qdrant_client import AsyncQdrantClient
from app.llm.openai_compat import OpenAICompatClient
from app.api.schemas.openai import ChatMessage, RetrievalDocument, RetrievedContext


DEFAULT_TOP_K = 15
DEFAULT_SCORE_THRESHOLD = 0.5
DEFAULT_CONTEXT_MAX_CHARS = 12000
DEFAULT_PAYLOAD_TEXT_KEYS = ("text", "content", "chunk", "page_content") # TODO: Revisar esto, creo que la clave está aquí


@dataclass(frozen=True)
class KnowledgeSource:
    id: str
    name: str
    description: str
    collection: str
    vector_name: str | None = None


DEFAULT_SEARCH_SOURCES = (
    KnowledgeSource(
        id="devices",
        name="Devices",
        description="Información sobre los dispositivos de la empresa, tales como servidores y equipos de usuario y de planta.",
        collection="devices",
    ),
    KnowledgeSource(
        id="employees",
        name="Employees",
        description="Información sobre los empleados de la empresa y su contacto corporativo, como correo electronico, telefono y extension.",
        collection="employees",
    ),
)



logger = logging.getLogger(__name__)


class QdrantRetriever():
    def __init__(self, settings: Settings, embedding_client: OpenAICompatClient) -> None:
        self.settings = settings
        self.embedding_client = embedding_client
        self.client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
        self.sources = {source.id: source for source in DEFAULT_SEARCH_SOURCES}

    async def retrieve(self, query: str, messages: list[ChatMessage]) -> RetrievedContext:
        return await self.retrieve_from_sources(query=query, messages=messages, source_ids=None)

    async def retrieve_from_sources(self, query: str, messages: list[ChatMessage], source_ids: list[str] | None = None) -> RetrievedContext:
        # Reescribe la consulta, basándose en el histórico, no solo en la última pregunta
        rewritten_query = self._rewrite_query(query, messages)
        logger.debug(
            "Starting retrieval from sources | query=%s rewritten_query=%s source_ids=%s",
            query,
            rewritten_query,
            source_ids,
        )
        # Generamos el embedding
        query_vector = await self.embedding_client.create_embedding(
            input_text=rewritten_query,
            model=self.settings.embedding_model,
        )
        # Preparamos los documentos y resolvemos las colecciones donde buscar 
        documents: list[RetrievalDocument] = []
        selected_sources = self._resolve_sources(source_ids)
        for source in selected_sources:
            logger.debug(
                "Querying Qdrant source | source_id=%s collection=%s vector_name=%s",
                source.id,
                source.collection,
                source.vector_name,
            )
            # Para cada posible fuente/colección
            search_kwargs: dict[str, Any] = {
                "collection_name": source.collection,
                "query": query_vector,
                "limit": DEFAULT_TOP_K,
                "with_payload": True,
                "score_threshold": DEFAULT_SCORE_THRESHOLD,
            }
            if source.vector_name:
                logger.debug(
                    "Using named vector for Qdrant search | source=%s collection=%s vector_name=%s",
                    source.id,
                    source.collection,
                    source.vector_name,
                )
                search_kwargs["using"] = source.vector_name
            results = await self.client.query_points(**search_kwargs)
            points = results.points if hasattr(results, "points") else []
            logger.debug("Qdrant source response | source_id=%s points_found=%d", source.id, len(points))
            documents.extend(self._point_to_document(point, source) for point in points)
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
                for document in documents
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
        # TODO: Revisar esto, creo que estaría mejor pedirle al modelo que resuma la intención
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
        # TODO: Revisar esto
        payload = dict(point.payload or {})
        text = self._extract_text(payload)
        metadata = self._extract_metadata(payload, source)
        return RetrievalDocument(
            id=f"{source.id}:{point.id}",
            score=float(point.score or 0.0),
            text=text,
            metadata=metadata,
        )

    def _extract_text(self, payload: dict[str, Any]) -> str:
        # TODO: Revisar esto, OJO!!!
        for key in DEFAULT_PAYLOAD_TEXT_KEYS:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _extract_metadata(self, payload: dict[str, Any], source: KnowledgeSource) -> dict[str, Any]:
        raw_metadata = payload.get("metadata")
        metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
        for key, value in payload.items():
            if key in {"content", "metadata"}:
                continue
            metadata.setdefault(key, value)
        metadata.setdefault("collection", source.collection)
        metadata.setdefault("source_id", source.id)
        metadata.setdefault("source_name", source.name)
        return metadata

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

    def _resolve_sources(self, source_ids: list[str] | None) -> list[KnowledgeSource]:
        if not source_ids:
            return list(self.sources.values())
        unknown = [source_id for source_id in source_ids if source_id not in self.sources]
        if unknown:
            available = ", ".join(sorted(self.sources))
            raise ValueError(f"Unknown knowledge sources: {', '.join(unknown)}. Available: {available}")
        return [self.sources[source_id] for source_id in source_ids]
