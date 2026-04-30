from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.dependencies.auth import get_current_user
from app.db.models.user import User
from app.schemas.discord_webhook import (
    DiscordWebhookTestRequest,
    DiscordWebhookTestResponse,
)
from app.services.discord_webhook import send_discord_message

router = APIRouter(prefix="/tools", tags=["tools"])


@router.post(
    "/test-discord-webhook",
    response_model=DiscordWebhookTestResponse,
    status_code=status.HTTP_200_OK,
)
async def test_discord_webhook_route(
    payload: DiscordWebhookTestRequest,
    request: Request,
    _current_user: User = Depends(get_current_user),
) -> DiscordWebhookTestResponse:
    http_client = request.app.state.resources.get("http_client")
    if http_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Shared HTTP client is not available.",
        )

    await send_discord_message(
        http_client,
        request.app.state.settings,
        message=payload.message,
    )

    return DiscordWebhookTestResponse(
        delivered=True,
        message_preview=payload.message[:120],
    )

