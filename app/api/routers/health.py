from fastapi import APIRouter


router = APIRouter(tags=["Health"])


@router.get(
    path="/health",
    summary="Comprueba el estado de la aplicación",
    description="Comprueba el estado de la aplicación",
)
async def healthcheck() -> dict[str, str]:
    return { "status": "ok" }
