from functools import lru_cache
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="ragent", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    chat_provider: str = Field(alias="CHAT_PROVIDER")
    chat_api_key: str = Field(default="", alias="CHAT_API_KEY")
    chat_base_url: str = Field(alias="CHAT_BASE_URL")
    chat_model: str = Field(alias="CHAT_MODEL")

    embedding_provider: str = Field(alias="EMBEDDING_PROVIDER")
    embedding_api_key: str = Field(default="", alias="EMBEDDING_API_KEY")
    embedding_base_url: str = Field(alias="EMBEDDING_BASE_URL")
    embedding_model: str = Field(alias="EMBEDDING_MODEL")

    llm_timeout_seconds: float = Field(default=60.0, alias="LLM_TIMEOUT_SECONDS")
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=800, alias="LLM_MAX_TOKENS")

    qdrant_url: str = Field(alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")

    @model_validator(mode="after")
    def validate_required_values(self) -> "Settings":
        required_non_empty = {
            "CHAT_PROVIDER": self.chat_provider,
            "CHAT_BASE_URL": self.chat_base_url,
            "CHAT_MODEL": self.chat_model,
            "EMBEDDING_PROVIDER": self.embedding_provider,
            "EMBEDDING_BASE_URL": self.embedding_base_url,
            "EMBEDDING_MODEL": self.embedding_model,
            "QDRANT_URL": self.qdrant_url,
        }
        missing = [name for name, value in required_non_empty.items() if not str(value).strip()]
        if missing:
            formatted = ", ".join(missing)
            raise ValueError(
                f"Missing required settings: {formatted}. "
                "Define them in .env or the process environment before starting the app."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
