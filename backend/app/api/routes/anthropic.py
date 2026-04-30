from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.dependencies.auth import get_current_user
from app.db.models.user import User
from app.services.claude import list_anthropic_models

router = APIRouter(prefix="/tools", tags=["tools"])


@router.get(
    "/anthropic-models",
    status_code=status.HTTP_200_OK,
)
async def list_anthropic_models_route(
    request: Request,
    _current_user: User = Depends(get_current_user),
) -> dict:
    http_client = request.app.state.resources.get("http_client")
    if http_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Shared HTTP client is not available.",
        )

    return await list_anthropic_models(
        http_client,
        request.app.state.settings,
    )
