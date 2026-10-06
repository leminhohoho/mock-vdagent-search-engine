"""Runtime settings from `.env` and environment variables (`.env` wins)."""

from datetime import datetime
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    embedding_base_url: str = Field(
        "https://api.openai.com/v1",
        validation_alias=AliasChoices("EMBEDDING_BASE_URL", "OPENAI_BASE_URL"),
    )
    embedding_api_key: str | None = Field(
        None, validation_alias=AliasChoices("EMBEDDING_API_KEY", "OPENAI_API_KEY")
    )
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = Field(128, ge=1)

    corpus_dir: Path = Path("data/corpus")

    # Postgres connection string (Supabase session pooler); required by ingest and serve.
    database_url: str | None = None
    db_pool_size: int = Field(5, ge=1)

    mock_api_key: str | None = None
    mock_now: datetime | None = None

    host: str = "127.0.0.1"
    port: int = 8000

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # The project's .env beats the shell: a globally exported OPENAI_API_KEY for another
        # provider must not be paired with the .env base URL.
        return init_settings, dotenv_settings, env_settings, file_secret_settings
