from app.core.config import Settings
import httpx

DISCORD_MESSAGE_LIMIT = 1900


async def send_discord_message(
    http_client: httpx.AsyncClient,
    settings: Settings,
    *,
    message: str,
) -> None:
    if not settings.discord_webhook_url:
        raise RuntimeError("Discord webhook URL is not configured.")

    response = await http_client.post(
        settings.discord_webhook_url,
        json={
            "username": settings.discord_webhook_username,
            "content": message,
        },
        timeout=settings.discord_webhook_timeout_seconds,
    )
    response.raise_for_status()


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
