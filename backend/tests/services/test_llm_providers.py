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

    # usage_metadata=None matches a real (if unusual) SDK response shape -
    # complete() must not crash reading token counts off of it.
    fake_response = type(
        "FakeResponse", (), {"text": "hello from gemini", "usage_metadata": None}
    )()
    with patch.object(
        provider._client.aio.models, "generate_content", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = fake_response
        result = await provider.complete([{"role": "user", "content": "hi"}])

    assert result == "hello from gemini"
    _, call_kwargs = mock_generate.call_args
    assert call_kwargs["model"] == settings.gemini_model


@pytest.mark.asyncio(loop_scope="session")
async def test_gemini_provider_logs_token_usage(caplog):
    """Priority 6/7 coverage: token/cost logging in the LLM provider layer."""
    settings = Settings(gemini_api_key="test-key", llm_provider="gemini", gemini_model="gemma-4-26b-a4b-it")
    provider = GeminiProvider(settings)

    fake_usage = type(
        "FakeUsage",
        (),
        {"prompt_token_count": 12, "candidates_token_count": 34, "thoughts_token_count": 56},
    )()
    fake_response = type(
        "FakeResponse", (), {"text": "hello", "usage_metadata": fake_usage}
    )()

    with patch.object(
        provider._client.aio.models, "generate_content", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = fake_response
        with caplog.at_level("INFO", logger="app.llm_usage"):
            await provider.complete([{"role": "user", "content": "hi"}])

    records = [r for r in caplog.records if r.name == "app.llm_usage"]
    assert len(records) == 1
    assert records[0].input_tokens == 12
    assert records[0].output_tokens == 34
    assert records[0].thinking_tokens == 56
    assert records[0].llm_model == "gemma-4-26b-a4b-it"
    # gemma-4-26b-a4b-it is priced at (0.0, 0.0) - free tier, confirmed live.
    assert records[0].estimated_cost_usd == 0.0


@pytest.mark.asyncio(loop_scope="session")
async def test_anthropic_provider_logs_token_usage(caplog):
    """Priority 6/7 coverage: token/cost logging in the LLM provider layer."""
    settings = Settings(anthropic_api_key="test-key", anthropic_model="claude-haiku-4-5")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "hi there"}],
                "usage": {"input_tokens": 20, "output_tokens": 5},
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = AnthropicProvider(settings, http_client=client)
        with caplog.at_level("INFO", logger="app.llm_usage"):
            await provider.complete([{"role": "user", "content": "hi"}])

    records = [r for r in caplog.records if r.name == "app.llm_usage"]
    assert len(records) == 1
    assert records[0].input_tokens == 20
    assert records[0].output_tokens == 5
    # claude-haiku-4-5 is priced at (1.00, 5.00) USD/MTok - (20*1.00 + 5*5.00) / 1_000_000.
    assert records[0].estimated_cost_usd == pytest.approx((20 * 1.00 + 5 * 5.00) / 1_000_000)


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
