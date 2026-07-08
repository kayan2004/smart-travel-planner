import time
from functools import lru_cache

from google import genai
from google.genai import types

from app.core.config import Settings
from app.services.llm_providers.protocol import Message, split_system_and_user
from app.services.llm_providers.usage_logging import log_completion_usage


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

        started_at = time.monotonic()
        response = await self._client.aio.models.generate_content(
            model=model,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system or None,
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        elapsed_seconds = time.monotonic() - started_at

        usage = response.usage_metadata
        thinking_tokens = (usage.thoughts_token_count or 0) if usage else 0
        log_completion_usage(
            provider="gemini",
            model=model,
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=(usage.candidates_token_count or 0) if usage else 0,
            latency_seconds=elapsed_seconds,
            # Gemma 4 spends a real, sometimes-large token budget on internal
            # "thinking" before the visible answer (see gemini_max_tokens's
            # comment in app/core/config.py) - tracked separately from
            # output_tokens since it's diagnostic, not part of the answer.
            extra_tokens={"thinking_tokens": thinking_tokens} if thinking_tokens else None,
        )
        return (response.text or "").strip()
