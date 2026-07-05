import httpx

from app.core.config import Settings
from app.services.llm_providers.anthropic_provider import AnthropicProvider
from app.services.llm_providers.gemini_provider import GeminiProvider
from app.services.llm_providers.protocol import LLMProvider


def get_llm_provider(settings: Settings, *, http_client: httpx.AsyncClient) -> LLMProvider:
    if settings.llm_provider == "gemini":
        return GeminiProvider(settings)
    if settings.llm_provider == "anthropic":
        return AnthropicProvider(settings, http_client=http_client)
    raise RuntimeError(
        f"Unknown llm_provider '{settings.llm_provider}' - expected 'anthropic' or 'gemini'."
    )
