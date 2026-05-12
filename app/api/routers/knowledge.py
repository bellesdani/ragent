from typing import Any, List
from dataclasses import asdict
from pydantic import ValidationError
from fastapi.responses import JSONResponse
from fastapi import APIRouter, Depends, File, Request, UploadFile
from app.core.knowledge_source.service import KnowledgeSourceService
from app.core.knowledge_source.entities import KnowledgeSourceDefinition
from app.api.schemas.knowledge import (
    KnowledgeSourceCreateResponse,
    KnowledgeSourceUpsertResponse,
    KnowledgeSourceErrorResponse,
    KnowledgeSourcesListResponse,
    KnowledgeSourceUpsertResult,
    KnowledgeSourceCreateResult,
    KnowledgeSourcesListResult,
    KnowledgeSourceError,
    KnowledgeSourceItem,
)


router = APIRouter(tags=["Knowledge Sources"])


def get_knowledge_service(request: Request) -> KnowledgeSourceService:
    return request.app.state.knowledge_service


@router.get("/knowledge-source", response_model=KnowledgeSourcesListResponse)
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


@router.post("/knowledge-source/{knowledge_source_id}", response_model=KnowledgeSourceCreateResponse)
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


@router.post("/knowledge-source/{knowledge_source_id}/points/from-json", response_model=KnowledgeSourceUpsertResponse)
async def upsert_points_in_knowledge_source_from_json(
    knowledge_source_id: str,
    data: object | None = None,
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


@router.post("/knowledge-source/{knowledge_source_id}/points/from-html", response_model=KnowledgeSourceUpsertResponse)
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


def _build_knowledge_upsert_result(result: dict[str, Any]) -> KnowledgeSourceUpsertResult:
    points = result.get("points", 0)
    if not isinstance(points, int):
        raise ValueError("El resultado de la ingesta no es valido")
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
            message="Los datos enviados no son validos",
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
            message="La fuente de conocimiento no esta correctamente configurada",
        )
    return _build_error_response(
        status_code=500,
        code="internal_error",
        message="Se ha producido un error interno procesando la operacion",
    )


def _build_error_response(
    status_code: int,
    code: str,
    message: str,
    details: Any | None = None,
) -> JSONResponse:
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
