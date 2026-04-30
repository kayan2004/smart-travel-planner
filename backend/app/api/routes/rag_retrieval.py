from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import get_current_user
from app.db.dependencies import get_db_session
from app.db.models.user import User
from app.schemas.rag_retrieval import RagRetrievalRequest, RagRetrievalResponse
from app.services.rag_retrieval import retrieve_destination_context

router = APIRouter(prefix="/tools", tags=["tools"])


@router.post(
    "/retrieve-destination-context",
    response_model=RagRetrievalResponse,
    status_code=status.HTTP_200_OK,
)
async def retrieve_destination_context_route(
    payload: RagRetrievalRequest,
    request: Request,
    _current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> RagRetrievalResponse:
    http_client = request.app.state.resources.get("http_client")
    if http_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Shared HTTP client is not available.",
        )

    return await retrieve_destination_context(
        session,
        http_client,
        request.app.state.settings,
        payload,
    )
