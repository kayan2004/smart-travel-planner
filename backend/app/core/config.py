from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

# The exact placeholder text also committed in .env.example - anyone can read
# it from this public repo, so it must never actually be usable as a real
# secret. Kept as a named constant (not inlined twice) so the field default
# below and the validator that rejects it can never drift out of sync.
_INSECURE_DEFAULT_JWT_SECRET = "change-this-development-secret-to-32-plus-chars"

# Every nested settings group below is itself a BaseSettings reading the same
# shared .env file, filtered by its own env_prefix - so e.g. RagSettings only
# ever sees RAG_* env vars. This keeps env var names identical to before this
# refactor (RAG_CHUNK_SIZE still means the same thing); only the Python access
# path changed, from settings.rag_chunk_size to settings.rag.chunk_size. A
# field whose original env var didn't share its group's prefix (frontend_origin,
# the two open_meteo_* weather URLs) keeps working via an explicit
# validation_alias instead of the prefix convention.


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="APP_", env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    name: str = "Smart Travel Planner API"
    env: str = "development"
    # Secure by default: a deployment that forgets to configure this
    # explicitly gets stack-trace-free error responses, not the reverse.
    # Local dev that wants full tracebacks sets APP_DEBUG=true in .env.
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    frontend_origin: str = Field(
        default="http://localhost:5173", validation_alias="FRONTEND_ORIGIN"
    )


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DATABASE_", env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ground_trip"
    echo: bool = False


class AuthSettings(BaseSettings):
    """No shared env-var prefix across these three (JWT_* vs ACCESS_TOKEN_*),
    so this group reads the original full names directly rather than via
    env_prefix."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    jwt_secret_key: str = _INSECURE_DEFAULT_JWT_SECRET
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    @field_validator("jwt_secret_key")
    @classmethod
    def _reject_insecure_jwt_secret(cls, value: str) -> str:
        # Fails loudly at startup (Settings() construction time) rather
        # than silently booting with a forgeable-JWT secret - this field
        # having *any* default at all is dev convenience, not a value
        # that's ever safe to actually run with. Unconditional (not gated
        # on APP_DEBUG/APP_ENV) on purpose: gating it on another
        # insecure-by-default flag would just move the same problem one
        # level up. Every environment, including CI, must set a real
        # JWT_SECRET_KEY - see .github/workflows/ci.yml for the test-only
        # value used there.
        if value == _INSECURE_DEFAULT_JWT_SECRET:
            raise ValueError(
                "JWT_SECRET_KEY is still the placeholder value from .env.example - "
                "set a real secret via the JWT_SECRET_KEY environment variable "
                "before starting the app."
            )
        if len(value) < 32:
            raise ValueError(
                "JWT_SECRET_KEY must be at least 32 characters - "
                f"got {len(value)}."
            )
        return value


class RagSettings(BaseSettings):
    """RAG ingestion (Wikivoyage -> pgvector)."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    source_manifest_path: str = "data/rag_source_manifest.json"
    chunk_size: int = 800
    chunk_overlap: int = 120
    fetch_timeout_seconds: float = 20.0
    user_agent: str = "ground-trip-rag-ingestion/1.0"
    embedding_batch_size: int = 32
    embedding_max_request_tokens: int = 10000
    estimated_chars_per_token: int = 4


class VoyageSettings(BaseSettings):
    """Voyage AI (embeddings provider)."""

    model_config = SettingsConfigDict(
        env_prefix="VOYAGE_", env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    api_key: str = ""
    api_base_url: str = "https://api.voyageai.com/v1"
    embedding_model: str = "voyage-4-lite"
    embedding_dimension: int = 1024
    timeout_seconds: float = 30.0
    requests_per_minute: int = 3
    max_retries: int = 3


class AnthropicSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ANTHROPIC_", env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    api_key: str = ""
    api_base_url: str = "https://api.anthropic.com"
    api_version: str = "2023-06-01"
    # No fast/strong tiers - removed entirely (2026-07-06) in favor of a single
    # configured model per provider. Model name is a deliberate, reviewable
    # code decision (not an env var) - see CLAUDE.md's memory or the git log
    # for why; kept on the cheaper default since this provider is dormant
    # while LLM_PROVIDER=gemini.
    model: str = "claude-haiku-4-5"
    max_tokens: int = 700
    temperature: float = 0.2


class GeminiSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GEMINI_", env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    api_key: str = ""
    # No base_url/version settings here - the google-genai SDK resolves the
    # Gemini Developer API endpoint itself; unlike AnthropicProvider (plain
    # REST), there's no wire-level detail for us to configure.
    # gemini-3.1-flash-lite - a paid Gemini-branded model, not the free Gemma
    # tier. Confirmed live-working (2026-07-09): ~$0.001/run at this app's
    # prompt sizes, ~5x faster and ~9x cheaper than gemini-3.1-pro-preview in
    # a direct comparison, with equivalent answer quality in that test. Model
    # choice lives here (not .env) on purpose - it's a cost/quality decision,
    # not per-environment config; change it via a reviewed code edit, not a
    # silent runtime toggle. See backend/README.md's "Gemini (default
    # provider)" section for the full live-testing story and other
    # confirmed-working model names (gemma-4-26b-a4b-it is the free-tier
    # fallback if billing isn't set up).
    model: str = "gemini-3.1-flash-lite"
    # Gemma 4 spends a substantial chunk of max_output_tokens on internal
    # "thinking" tokens (thought=True response parts) before ever emitting
    # the actual answer - confirmed live: a trivial prompt used ~1500
    # thinking tokens before the real ~85-token answer. A low budget (this
    # was 700) truncates mid-thought (finish_reason=MAX_TOKENS) and
    # response.text comes back empty. Kept generous so switching back to the
    # free Gemma fallback doesn't silently truncate.
    max_tokens: int = 4096
    temperature: float = 0.2


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPENAI_", env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    api_key: str = ""
    api_base_url: str = "https://api.openai.com/v1"
    # gpt-5.4-nano: cheapest OpenAI chat-completions tier ($0.20/$1.25 per
    # MTok in/out), cross-checked against OpenAI's own pricing docs and
    # independent aggregators 2026-07-13 - NOT live-call-verified, since
    # this repo has no OPENAI_API_KEY configured (unlike gemini_model,
    # which was confirmed via a real client.models.list() call). Verify
    # live before treating this as trustworthy the way the Gemini default
    # is - see llm_providers/openai_provider.py's max_completion_tokens
    # note for the other unverified detail.
    model: str = "gpt-5.4-nano"
    max_tokens: int = 700
    temperature: float = 0.2


class WeatherSettings(BaseSettings):
    """Live weather (Open-Meteo). The two base URLs kept their original
    OPEN_METEO_* env var names (predates this settings group) via an explicit
    alias; only the timeout was ever WEATHER_-prefixed."""

    model_config = SettingsConfigDict(
        env_prefix="WEATHER_", env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    geocoding_base_url: str = Field(
        default="https://geocoding-api.open-meteo.com",
        validation_alias="OPEN_METEO_GEOCODING_BASE_URL",
    )
    forecast_base_url: str = Field(
        default="https://api.open-meteo.com",
        validation_alias="OPEN_METEO_FORECAST_BASE_URL",
    )
    request_timeout_seconds: float = 20.0


class DestinationSettings(BaseSettings):
    """Destination corpus ingestion (backend/scripts/ingest_destinations.py)."""

    model_config = SettingsConfigDict(
        env_prefix="DESTINATION_", env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    seed_manifest_path: str = "data/destination_seed_manifest.json"
    embedding_version: str = "v1"
    fetch_timeout_seconds: float = 20.0
    user_agent: str = (
        "ground-trip-destination-ingestion/1.0 "
        "(contact: kayanabdepbaki@gmail.com)"
    )
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0


class OpenTripMapSettings(BaseSettings):
    """POI enrichment during destination ingestion - optional, degrades
    gracefully if api_key is blank."""

    model_config = SettingsConfigDict(
        env_prefix="OPENTRIPMAP_", env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    api_key: str = ""
    base_url: str = "https://api.opentripmap.com/0.1/en"
    radius_meters: int = 20000
    poi_limit: int = 100


class Settings(BaseSettings):
    app: AppSettings = Field(default_factory=AppSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    rag: RagSettings = Field(default_factory=RagSettings)
    voyage: VoyageSettings = Field(default_factory=VoyageSettings)

    # "anthropic" | "gemini" | "openai" - selects the provider for all three
    # LLM call sites (extraction, synthesis, cluster naming) when no
    # per-request BYOK override is present. See app/services/llm_providers/
    # and app/core/byok.py (BYOK requests get a request-scoped copy of this
    # Settings object with the override applied, never a mutation of the
    # shared singleton).
    llm_provider: str = "gemini"
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)

    # On by default: recommend_destinations() re-ranks the cosine-retrieved
    # slate with the LightGBM ranker whenever artifacts/ranker/model.joblib
    # exists (falls back to cosine order if it doesn't). The shipped model is
    # still trained on a synthetic cold-start bootstrap, not real feedback -
    # see backend/README.md's "Learning-to-Rank" section for that caveat and
    # the real-feedback retrain path (scripts/train_ranker.py retrain).
    ranker_enabled: bool = True

    # Free-tier abuse protection for a public deploy paying for LLM calls on
    # the SERVER's key. A request that brings its own key (BYOK) bypasses
    # both of these entirely - the user is paying, so there's nothing to
    # protect. Enforced in app/api/routes/agent_runs.py against persisted
    # agent_runs rows (survives restart/redeploy, unlike the in-memory rate
    # limiter). See backend/README.md's "Server-Key Free Tier" section.
    #  - Per account, lifetime: after this many server-key runs the account
    #    must switch to BYOK to keep planning.
    free_server_runs_per_account: int = 1
    #  - Global, per UTC calendar month: once the summed estimated cost of
    #    all server-key runs this month reaches this ceiling, EVERY user
    #    falls back to BYOK until the month rolls over. A coarse backstop
    #    against multi-account signup abuse (which the per-account cap alone
    #    can't stop). Estimated cost, not the real Google bill - see
    #    app/services/llm_providers/usage_logging.py's pricing table.
    server_key_monthly_budget_usd: float = 1.0

    weather: WeatherSettings = Field(default_factory=WeatherSettings)
    destination: DestinationSettings = Field(default_factory=DestinationSettings)
    opentripmap: OpenTripMapSettings = Field(default_factory=OpenTripMapSettings)

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
