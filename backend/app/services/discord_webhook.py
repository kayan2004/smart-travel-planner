import asyncio

from app.core.config import Settings
import httpx

DISCORD_MESSAGE_LIMIT = 1900


async def send_discord_message(
    http_client: httpx.AsyncClient,
    settings: Settings,
    *,
    message: str,
) -> None:
    """Retries on transient failures - a 429 (rate limited, honoring Discord's
    own Retry-After header when present), a 5xx, or a network-level error
    (connection refused, timeout, DNS failure). Does NOT retry a 4xx other
    than 429 (bad/deleted webhook URL, malformed payload) - retrying a
    permanently broken webhook just wastes the retry budget, same reasoning
    Voyage's own retry loop uses (app/services/voyage_embeddings.py) for its
    real vs. rate-limit failure split.
    """
    if not settings.discord_webhook_url:
        raise RuntimeError("Discord webhook URL is not configured.")

    payload = {
        "username": settings.discord_webhook_username,
        "content": message,
    }
    last_error: Exception | None = None

    for attempt in range(settings.discord_webhook_max_retries + 1):
        try:
            response = await http_client.post(
                settings.discord_webhook_url,
                json=payload,
                timeout=settings.discord_webhook_timeout_seconds,
            )
        except httpx.TransportError as exc:
            last_error = exc
            if attempt < settings.discord_webhook_max_retries:
                await asyncio.sleep(_retry_delay_seconds(None, settings, attempt))
                continue
            break

        if attempt < settings.discord_webhook_max_retries and (
            response.status_code == 429 or response.status_code >= 500
        ):
            await asyncio.sleep(_retry_delay_seconds(response, settings, attempt))
            continue

        response.raise_for_status()
        return

    if last_error is not None:
        raise last_error
    raise RuntimeError(
        f"Discord webhook delivery failed after {settings.discord_webhook_max_retries} retries."
    )


def _retry_delay_seconds(
    response: httpx.Response | None,
    settings: Settings,
    attempt: int,
) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(float(retry_after), 0.5)
            except ValueError:
                pass
    return settings.discord_webhook_retry_backoff_seconds * (2**attempt)


async def send_trip_plan_to_discord(
    http_client: httpx.AsyncClient,
    settings: Settings,
    *,
    user_email: str,
    prompt: str,
    response_text: str,
    status: str,
) -> None:
    message = _format_trip_plan_message(
        user_email=user_email,
        prompt=prompt,
        response_text=response_text,
        status=status,
    )
    await send_discord_message(
        http_client,
        settings,
        message=message[:DISCORD_MESSAGE_LIMIT],
    )


def _format_trip_plan_message(
    *,
    user_email: str,
    prompt: str,
    response_text: str,
    status: str,
) -> str:
    safe_prompt = prompt.strip().replace("\n", " ")
    header = (
        "## Smart Travel Assistant Recommendation\n"
        f"**User:** {user_email}\n"
        f"**Run status:** {status}\n"
    )
    prompt_block = f"**Request:** {safe_prompt}\n\n"
    plan_block = f"**Recommended plan:**\n{response_text.strip()}"
    return header + prompt_block + plan_block
