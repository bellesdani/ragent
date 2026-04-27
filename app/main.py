from fastapi import FastAPI

from app.api.routes import create_router
from app.core.chat import build_chat_service
from app.core.config import configure_logging, get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    chat_service = build_chat_service(settings)

    app = FastAPI(title=settings.app_name)
    app.state.settings = settings
    app.state.chat_service = chat_service
    app.include_router(create_router(settings=settings, chat_service=chat_service))
    return app


app = create_app()
