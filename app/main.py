from fastapi import FastAPI

from app.core.config import get_settings
from app.api.routes import create_router
from app.core.chat import build_chat_service


def create_app() -> FastAPI:
    settings = get_settings()
    chat_service = build_chat_service(settings)

    app = FastAPI(title="Agents")
    app.state.settings = settings
    app.state.chat_service = chat_service
    app.include_router(create_router(settings=settings, chat_service=chat_service))
    return app


app = create_app()
