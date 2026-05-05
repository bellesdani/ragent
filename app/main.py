from fastapi import FastAPI

from app.api.routers import chat, health
from app.core.config import get_settings
from app.core.chat import build_chat_service


def create_app() -> FastAPI:
    settings = get_settings()
    chat_service = build_chat_service(settings)

    app = FastAPI(title="Agents")
    app.state.settings = settings
    app.state.chat_service = chat_service
    app.include_router(health.router)
    app.include_router(chat.router)
    return app


app = create_app()
