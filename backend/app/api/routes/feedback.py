from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.rate_limit import feedback_ip_rate_limiter
from app.db.dependencies import get_db_session
from app.schemas.feedback import FeedbackCreate, FeedbackRead
from app.services.feedback import RecommendationNotFoundError, submit_feedback

router = APIRouter(tags=["feedback"])


@router.post("/feedback", response_model=FeedbackRead, status_code=status.HTTP_200_OK)
async def submit_feedback_route(
    payload: FeedbackCreate,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> FeedbackRead:
    client_ip = request.client.host if request.client else "unknown"
    if not await feedback_ip_rate_limiter.check(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests - please slow down.")

    try:
        return await submit_feedback(session, payload)
    except RecommendationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
