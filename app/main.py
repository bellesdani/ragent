from fastapi import FastAPI
from app.config.config import get_settings
from app.config.logging import configure_logging
from app.api.routes.chat import router as chat_router


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(title=settings.app_name)
    app.include_router(chat_router)
    return app


app = create_app()
