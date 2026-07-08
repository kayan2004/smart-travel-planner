"""Coverage for app/services/discord_webhook.py's retry-with-backoff (audit
item 8 - this was previously fire-once, no retry).
"""

import httpx
import pytest

from app.core.config import Settings
from app.services.discord_webhook import send_discord_message


def _settings(**overrides: object) -> Settings:
    return Settings(
        discord_webhook_url="https://discord.com/api/webhooks/test/token",
        discord_webhook_retry_backoff_seconds=0.01,  # keep the test fast
        **overrides,
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_succeeds_on_first_try_without_retrying():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(204)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await send_discord_message(client, _settings(), message="hi")

    assert call_count == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_retries_on_429_then_succeeds():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(429, headers={"Retry-After": "0.01"})
        return httpx.Response(204)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await send_discord_message(client, _settings(), message="hi")

    assert call_count == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_retries_on_500_then_succeeds():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return httpx.Response(503)
        return httpx.Response(204)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await send_discord_message(client, _settings(discord_webhook_max_retries=3), message="hi")

    assert call_count == 3


@pytest.mark.asyncio(loop_scope="session")
async def test_does_not_retry_a_permanent_client_error():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(400, json={"message": "Invalid webhook payload"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await send_discord_message(client, _settings(), message="hi")

    assert call_count == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_raises_after_exhausting_retries_on_persistent_500():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(500)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await send_discord_message(
                client, _settings(discord_webhook_max_retries=2), message="hi"
            )

    assert call_count == 3  # initial attempt + 2 retries


@pytest.mark.asyncio(loop_scope="session")
async def test_retries_on_network_error_then_succeeds():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.ConnectError("simulated connection failure", request=request)
        return httpx.Response(204)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await send_discord_message(client, _settings(), message="hi")

    assert call_count == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_raises_for_missing_webhook_url():
    async with httpx.AsyncClient() as client:
        with pytest.raises(RuntimeError, match="not configured"):
            await send_discord_message(client, Settings(discord_webhook_url=""), message="hi")
