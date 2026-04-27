from __future__ import annotations

import logging
from logging.config import dictConfig


APP_LOGGER_NAME = "app"
LOG_FORMAT = "%(levelname)s: [%(name)s] %(message)s"
NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "httpcore.http11",
    "openai",
    "openai._base_client",
    "pydantic_ai",
    "qdrant_client",
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
)


class AppDebugFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        is_app_log = record.name == APP_LOGGER_NAME or record.name.startswith(f"{APP_LOGGER_NAME}.")
        return is_app_log or record.levelno >= logging.WARNING


def configure_logging(log_level: str) -> None:
    level_name = (log_level or "INFO").upper()
    app_level = logging.getLevelName(level_name)
    if not isinstance(app_level, int):
        app_level = logging.INFO

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {
                "app_debug_filter": {
                    "()": AppDebugFilter,
                },
            },
            "formatters": {
                "default": {
                    "format": LOG_FORMAT,
                },
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "level": "DEBUG",
                    "formatter": "default",
                    "filters": ["app_debug_filter"],
                },
            },
            "root": {
                "level": "WARNING",
                "handlers": ["default"],
            },
            "loggers": {
                APP_LOGGER_NAME: {
                    "level": app_level,
                    "handlers": [],
                    "propagate": True,
                },
                **{
                    logger_name: {
                        "level": "WARNING",
                        "handlers": [],
                        "propagate": True,
                    }
                    for logger_name in NOISY_LOGGERS
                },
            },
        }
    )
