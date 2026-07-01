from typing import Any, List
from dataclasses import asdict
from pydantic import ValidationError
from fastapi.responses import JSONResponse
from app.core.knowledge_source.service import KnowledgeSourceService
from fastapi import APIRouter, Body, Depends, File, Request, UploadFile
from app.core.knowledge_source.entities import KnowledgeSourceDefinition
from app.api.schemas.knowledge import (
    KnowledgeSourceCreateResponse,
    KnowledgeSourceUpsertResponse,
    KnowledgeSourceSearchResponse,
    KnowledgeSourceSearchRequest,
    KnowledgeSourceErrorResponse,
    KnowledgeSourcesListResponse,
    KnowledgeSourceUpsertResult,
    KnowledgeSourceSearchResult,
    KnowledgeSourceCreateResult,
    KnowledgeSourcesListResult,
    KnowledgeSourceSearchItem,
    KnowledgeSourceError,
    KnowledgeSourceItem,
)


KNOWLEDGE_MANAGEMENT_TAG = "Knowledge Sources - Management"
KNOWLEDGE_INGESTION_TAG = "Knowledge Sources - Ingestion"
KNOWLEDGE_SEARCH_TAG = "Knowledge Sources - Search"


router = APIRouter()


def get_knowledge_service(request: Request) -> KnowledgeSourceService:
    return request.app.state.knowledge_service


@router.get(
    path="/knowledge-source",
    response_model=KnowledgeSourcesListResponse,
    tags=[KNOWLEDGE_MANAGEMENT_TAG],
    summary="Lista las fuentes de conocimiento",
    description="Lista las fuentes de conocimiento disponibles para ingesta y búsqueda de datos",
)
async def get_knowledge_sources(
    knowledge_service: KnowledgeSourceService = Depends(get_knowledge_service)
) -> KnowledgeSourcesListResponse | JSONResponse:
    try:
        knowledge_sources: List[KnowledgeSourceDefinition] = knowledge_service.list_knowledge_sources()
        items = [
            KnowledgeSourceItem.model_validate(asdict(knowledge_source))
            for knowledge_source in knowledge_sources
        ]
        return KnowledgeSourcesListResponse(
            result=KnowledgeSourcesListResult(
                items=items,
                count=len(items),
            )
        )
    except Exception as exc:
        return _build_knowledge_error_response(exc)


@router.post(
    path="/knowledge-source/{knowledge_source_id}",
    response_model=KnowledgeSourceCreateResponse,
    tags=[KNOWLEDGE_MANAGEMENT_TAG],
    summary="Crea una fuente de conocimiento",
    description="Crea una fuente de conocimiento (colección en Qdrant) utilizando un esquema previamente definido",
)
async def create_knowledge_source(
    knowledge_source_id: str,
    knowledge_service: KnowledgeSourceService = Depends(get_knowledge_service)
) -> KnowledgeSourceCreateResponse | JSONResponse:
    try:
        collection_created = await knowledge_service.create_knowledge_source(knowledge_source_id)
        return KnowledgeSourceCreateResponse(
            knowledge_source_id=knowledge_source_id,
            result=KnowledgeSourceCreateResult(
                collection_created=collection_created,
            ),
        )
    except Exception as exc:
        return _build_knowledge_error_response(exc)


@router.post(
    path="/knowledge-source/{knowledge_source_id}/points/from-json",
    response_model=KnowledgeSourceUpsertResponse,
    tags=[KNOWLEDGE_INGESTION_TAG],
    summary="Inserta datos estructurados a una fuente de conocimiento",
    description="Inserta o actualiza un punto de Qdrant de una fuente de conocimiento a partir de un JSON",
)
async def upsert_points_in_knowledge_source_from_json(
    knowledge_source_id: str,
    data: object | None = Body(default=None),
    knowledge_service: KnowledgeSourceService = Depends(get_knowledge_service),
) -> KnowledgeSourceUpsertResponse | JSONResponse:
    try:
        if not isinstance(data, list):
            raise ValueError("Los datos JSON deben enviarse como una lista de objetos")
        result = await knowledge_service.upsert_knowledge_source_data(knowledge_source_id, data)
        return KnowledgeSourceUpsertResponse(
            knowledge_source_id=knowledge_source_id,
            result=_build_knowledge_upsert_result(result),
        )
    except Exception as exc:
        return _build_knowledge_error_response(exc)


@router.post(
    path="/knowledge-source/{knowledge_source_id}/points/from-html",
    response_model=KnowledgeSourceUpsertResponse,
    tags=[KNOWLEDGE_INGESTION_TAG],
    summary="Inserta datos en HTML a una fuente de conocimiento",
    description="Inserta o actualiza un punto/documento de Qdrant de una fuente de conocimiento a partir de un documento HTML",
)
async def upsert_points_in_knowledge_source_from_html(
    knowledge_source_id: str, 
    file: UploadFile | None = File(default=None),
    knowledge_service: KnowledgeSourceService = Depends(get_knowledge_service),
) -> KnowledgeSourceUpsertResponse | JSONResponse:
    try:
        if file is None:
            raise ValueError("Debe adjuntarse un fichero HTML")
        result = await knowledge_service.upsert_knowledge_source_data(
            knowledge_source_id=knowledge_source_id,
            data={
                "filename": file.filename or "manual.html",
                "content": await file.read(),
                "content_type": file.content_type,
            }
        )
        return KnowledgeSourceUpsertResponse(
            knowledge_source_id=knowledge_source_id,
            result=_build_knowledge_upsert_result(result),
        )
    except Exception as exc:
        return _build_knowledge_error_response(exc)


@router.post(
    path="/knowledge-source/{knowledge_source_id}/search",
    response_model=KnowledgeSourceSearchResponse,
    tags=[KNOWLEDGE_SEARCH_TAG],
    summary="Búsqueda directa en una fuente de conocimiento",
    description="Ejecuta búsqueda directa en una fuente de conocimiento de Qdrant sin generar una respuesta conversacional",
)
async def search_knowledge_source(
    knowledge_source_id: str,
    search_request: KnowledgeSourceSearchRequest,
    knowledge_service: KnowledgeSourceService = Depends(get_knowledge_service),
) -> KnowledgeSourceSearchResponse | JSONResponse:
    try:
        retrieval = await knowledge_service.search_knowledge_source(
            knowledge_source_id=knowledge_source_id,
            query=search_request.query,
            limit=search_request.limit,
        )
        items = [
            KnowledgeSourceSearchItem.model_validate(document.model_dump())
            for document in retrieval.documents
        ]
        return KnowledgeSourceSearchResponse(
            knowledge_source_id=knowledge_source_id,
            result=KnowledgeSourceSearchResult(
                query=retrieval.query,
                last_data_update=retrieval.last_data_update,
                items=items,
                count=len(items),
            ),
        )
    except Exception as exc:
        return _build_knowledge_error_response(exc)


def _build_knowledge_upsert_result(result: dict[str, Any]) -> KnowledgeSourceUpsertResult:
    points = result.get("points", 0)
    if not isinstance(points, int):
        raise ValueError("El resultado de la ingesta no es válido")
    return KnowledgeSourceUpsertResult(
        points=points,
        summary={
            key: value
            for key, value in result.items()
            if key != "points"
        },
    )


def _build_knowledge_error_response(exc: Exception) -> JSONResponse:
    if isinstance(exc, ValidationError):
        return _build_error_response(
            status_code=422,
            code="invalid_knowledge_source_data",
            message="Los datos enviados no son válidos",
            details=exc.errors(
                include_url=False,
                include_context=False,
                include_input=False,
            ),
        )
    if isinstance(exc, ValueError):
        return _build_error_response(
            status_code=400,
            code="invalid_knowledge_source_request",
            message=str(exc),
        )
    if isinstance(exc, AssertionError):
        return _build_error_response(
            status_code=500,
            code="knowledge_source_configuration_error",
            message="La fuente de conocimiento no está correctamente configurada",
        )
    return _build_error_response(
        status_code=500,
        code="internal_error",
        message="Se ha producido un error interno procesando la operación",
    )


def _build_error_response(status_code: int, code: str, message: str, details: Any | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=KnowledgeSourceErrorResponse(
            error=KnowledgeSourceError(
                code=code,
                message=message,
                details=details,
            )
        ).model_dump(exclude_none=True),
    )
