from fastapi import FastAPI
from app.core.config import get_settings
from app.api.routes.chat import router as chat_router


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name)
    app.include_router(chat_router)
    return app


app = create_app()
