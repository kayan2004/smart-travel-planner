"""Priority 5 coverage: app/services/llm_providers/. All requests mocked -
no live Anthropic or Gemini call, ever. AnthropicProvider is REST-over-httpx
(mocked via httpx.MockTransport); GeminiProvider uses the google-genai SDK,
which doesn't go through httpx at all, so its client method is patched
directly instead (same technique used for live-Gemini-blocked verification
in an earlier session - see backend/README.md's "Provider-Agnostic LLM
Layer" section).

No fast/strong model tiers anymore (removed 2026-07-06) - complete() takes
no tier argument, each provider always uses its single configured model.
"""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.core.config import Settings
from app.services.llm_providers.anthropic_provider import AnthropicProvider
from app.services.llm_providers.factory import get_llm_provider
from app.services.llm_providers.gemini_provider import GeminiProvider


def _anthropic_mock_transport(response_text: str = "mocked response") -> httpx.MockTransport:
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": response_text}]},
        )

    transport = httpx.MockTransport(handler)
    transport.captured_requests = captured_requests  # type: ignore[attr-defined]
    return transport


@pytest.mark.asyncio(loop_scope="session")
async def test_anthropic_provider_builds_correct_request_and_parses_response():
    settings = Settings(anthropic_api_key="test-key", llm_provider="anthropic")
    transport = _anthropic_mock_transport("hello from anthropic")

    async with httpx.AsyncClient(transport=transport) as client:
        provider = AnthropicProvider(settings, http_client=client)
        result = await provider.complete(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hi."},
            ],
        )

    assert result == "hello from anthropic"
    request = transport.captured_requests[0]  # type: ignore[attr-defined]
    assert request.url.path == "/v1/messages"
    assert request.headers["x-api-key"] == "test-key"
    body = json.loads(request.content)
    assert body["model"] == settings.anthropic_model
    assert body["system"] == "You are a helpful assistant."
    assert body["messages"] == [{"role": "user", "content": "Say hi."}]


@pytest.mark.asyncio(loop_scope="session")
async def test_anthropic_provider_uses_the_single_configured_model():
    settings = Settings(anthropic_api_key="test-key", anthropic_model="claude-custom-test-model")
    transport = _anthropic_mock_transport()

    async with httpx.AsyncClient(transport=transport) as client:
        provider = AnthropicProvider(settings, http_client=client)
        await provider.complete([{"role": "user", "content": "hi"}])

    request = transport.captured_requests[0]  # type: ignore[attr-defined]
    body = json.loads(request.content)
    assert body["model"] == "claude-custom-test-model"


@pytest.mark.asyncio(loop_scope="session")
async def test_gemini_provider_parses_response_text():
    settings = Settings(gemini_api_key="test-key", llm_provider="gemini")
    provider = GeminiProvider(settings)

    fake_response = type("FakeResponse", (), {"text": "hello from gemini"})()
    with patch.object(
        provider._client.aio.models, "generate_content", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = fake_response
        result = await provider.complete([{"role": "user", "content": "hi"}])

    assert result == "hello from gemini"
    _, call_kwargs = mock_generate.call_args
    assert call_kwargs["model"] == settings.gemini_model


@pytest.mark.asyncio(loop_scope="session")
async def test_get_llm_provider_selects_by_config():
    settings_gemini = Settings(llm_provider="gemini", gemini_api_key="test-key")
    settings_anthropic = Settings(llm_provider="anthropic", anthropic_api_key="test-key")

    async with httpx.AsyncClient() as client:
        assert isinstance(get_llm_provider(settings_gemini, http_client=client), GeminiProvider)
        assert isinstance(
            get_llm_provider(settings_anthropic, http_client=client), AnthropicProvider
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_get_llm_provider_rejects_unknown_provider():
    settings = Settings(llm_provider="not-a-real-provider")
    async with httpx.AsyncClient() as client:
        with pytest.raises(RuntimeError):
            get_llm_provider(settings, http_client=client)
