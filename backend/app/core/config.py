from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    app_name: str = "Smart Travel Assistant API"
    app_env: str = "development"
    app_debug: bool = True
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    frontend_origin: str = "http://localhost:5173"
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
    llm_provider: str = "anthropic"  # "anthropic" | "gemini" - selects the provider for all
    # three LLM call sites (extraction, synthesis, cluster naming). See app/services/llm_providers.py.
    anthropic_api_key: str = ""
    anthropic_api_base_url: str = "https://api.anthropic.com"
    anthropic_api_version: str = "2023-06-01"
    anthropic_fast_model: str = "claude-3-5-haiku-latest"
    anthropic_strong_model: str = "claude-sonnet-4-5"
    anthropic_max_tokens: int = 700
    anthropic_temperature: float = 0.2
    gemini_api_key: str = ""
    gemini_api_base_url: str = "https://generativelanguage.googleapis.com"
    gemini_api_version: str = "v1beta"
    gemini_fast_model: str = "gemma-4-26b-a4b-it"
    gemini_strong_model: str = "gemma-4-31b-it"
    gemini_max_tokens: int = 700
    gemini_temperature: float = 0.2
    discord_webhook_url: str = ""
    discord_webhook_username: str = "Smart Travel Assistant"
    discord_webhook_timeout_seconds: float = 15.0
    open_meteo_geocoding_base_url: str = "https://geocoding-api.open-meteo.com"
    open_meteo_forecast_base_url: str = "https://api.open-meteo.com"
    weather_request_timeout_seconds: float = 20.0
    destination_seed_manifest_path: str = "data/destination_seed_manifest.json"
    destination_embedding_version: str = "v1"
    destination_fetch_timeout_seconds: float = 20.0
    destination_user_agent: str = (
        "smart-travel-assistant-destination-ingestion/1.0 "
        "(contact: kayanabdepbaki@gmail.com)"
    )
    destination_max_retries: int = 3
    destination_retry_backoff_seconds: float = 2.0
    opentripmap_api_key: str = ""
    opentripmap_base_url: str = "https://api.opentripmap.com/0.1/en"
    opentripmap_radius_meters: int = 20000
    opentripmap_poi_limit: int = 100
    numbeo_rankings_url: str = (
        "https://www.numbeo.com/cost-of-living/rankings_current.jsp"
    )

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
