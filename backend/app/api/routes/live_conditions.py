from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.dependencies.auth import get_current_user
from app.db.models.user import User
from app.schemas.live_conditions import LiveConditionsRequest, LiveConditionsResponse
from app.services.live_conditions import get_live_conditions

router = APIRouter(prefix="/tools", tags=["tools"])


@router.post(
    "/get-live-conditions",
    response_model=LiveConditionsResponse,
    status_code=status.HTTP_200_OK,
)
async def get_live_conditions_route(
    payload: LiveConditionsRequest,
    request: Request,
    _current_user: User = Depends(get_current_user),
) -> LiveConditionsResponse:
    http_client = request.app.state.resources.get("http_client")
    if http_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Shared HTTP client is not available.",
        )

    return await get_live_conditions(
        http_client,
        request.app.state.settings,
        payload,
    )

