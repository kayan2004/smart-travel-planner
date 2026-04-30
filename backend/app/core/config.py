from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Smart Travel Assistant API"
    app_env: str = "development"
    app_debug: bool = True
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/smart_travel_assistant"
    database_echo: bool = False
    jwt_secret_key: str = "change-this-development-secret-to-32-plus-chars"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    rag_source_manifest_path: str = "data/rag_source_manifest.json"
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 120
    rag_fetch_timeout_seconds: float = 20.0
    rag_user_agent: str = "smart-travel-assistant-rag-ingestion/1.0"
    rag_embedding_batch_size: int = 32
    rag_embedding_max_request_tokens: int = 10000
    rag_estimated_chars_per_token: int = 4
    voyage_api_key: str = ""
    voyage_api_base_url: str = "https://api.voyageai.com/v1"
    voyage_embedding_model: str = "voyage-4-lite"
    voyage_embedding_dimension: int = 1024
    voyage_timeout_seconds: float = 30.0
    voyage_requests_per_minute: int = 3
    voyage_max_retries: int = 3
    open_meteo_geocoding_base_url: str = "https://geocoding-api.open-meteo.com"
    open_meteo_forecast_base_url: str = "https://api.open-meteo.com"
    weather_request_timeout_seconds: float = 20.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
