import httpx

from app.core.config import Settings
from app.services.llm_providers.errors import raise_for_status_with_body
from app.services.llm_providers.protocol import Message, ModelTier, split_system_and_user


class AnthropicProvider:
    """Kept in place but unused while LLM_PROVIDER=gemini - selecting it back
    is a config value, not a code deletion. Pure REST over the shared
    httpx.AsyncClient, no Anthropic SDK dependency.
    """

    def __init__(self, settings: Settings, *, http_client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._http_client = http_client

    async def complete(
        self,
        messages: list[Message],
        model_tier: ModelTier,
        **opts: object,
    ) -> str:
        settings = self._settings
        if not settings.anthropic_api_key:
            raise RuntimeError("Anthropic API key is not configured.")

        model = (
            settings.anthropic_strong_model
            if model_tier == "strong"
            else settings.anthropic_fast_model
        )
        max_tokens = opts.get("max_tokens", settings.anthropic_max_tokens)
        temperature = opts.get("temperature", settings.anthropic_temperature)
        system, user_content = split_system_and_user(messages)

        response = await self._http_client.post(
            f"{settings.anthropic_api_base_url}/v1/messages",
            headers={
                "x-api-key": settings.anthropic_api_key,
                "anthropic-version": settings.anthropic_api_version,
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "messages": [{"role": "user", "content": user_content}],
            },
            timeout=settings.weather_request_timeout_seconds,
        )
        raise_for_status_with_body(
            response, context=f"Anthropic generation using model '{model}'"
        )
        payload = response.json()
        content_blocks = payload.get("content") or []
        text_parts = [
            block.get("text", "").strip()
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n\n".join(part for part in text_parts if part)
