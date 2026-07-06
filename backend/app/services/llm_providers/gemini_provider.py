from functools import lru_cache

from google import genai
from google.genai import types

from app.core.config import Settings
from app.services.llm_providers.protocol import Message, split_system_and_user


@lru_cache(maxsize=1)
def _get_client(api_key: str) -> genai.Client:
    # Memoized so the SDK's own transport (it does not accept the app's
    # shared httpx.AsyncClient) is constructed once per process, not once
    # per LLM call.
    return genai.Client(api_key=api_key)


class GeminiProvider:
    """Uses the google-genai SDK rather than raw REST. This SDK manages its
    own HTTP transport internally - it cannot reuse the shared
    httpx.AsyncClient the rest of this app is built around - so the client
    is memoized via _get_client() instead of constructed per request.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.gemini_api_key:
            raise RuntimeError("Gemini API key is not configured.")
        self._settings = settings
        self._client = _get_client(settings.gemini_api_key)

    async def complete(
        self,
        messages: list[Message],
        **opts: object,
    ) -> str:
        settings = self._settings
        model = settings.gemini_model
        max_tokens = opts.get("max_tokens", settings.gemini_max_tokens)
        temperature = opts.get("temperature", settings.gemini_temperature)
        system, user_content = split_system_and_user(messages)

        response = await self._client.aio.models.generate_content(
            model=model,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system or None,
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        return (response.text or "").strip()
