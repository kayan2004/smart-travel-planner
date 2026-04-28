from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import get_current_user
from app.core.security import create_access_token
from app.db.dependencies import get_db_session
from app.db.models.user import User
from app.schemas.auth import Token, UserCreate, UserLogin, UserRead
from app.services.auth import authenticate_user, create_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def signup(
    payload: UserCreate,
    session: AsyncSession = Depends(get_db_session),
) -> UserRead:
    user = await create_user(session, payload)
    return UserRead.model_validate(user)


@router.post("/login", response_model=Token, status_code=status.HTTP_200_OK)
async def login(
    payload: UserLogin,
    session: AsyncSession = Depends(get_db_session),
) -> Token:
    user = await authenticate_user(session, payload)
    token = create_access_token(user.email)
    return Token(access_token=token)


@router.get("/me", response_model=UserRead, status_code=status.HTTP_200_OK)
async def read_current_user(current_user: User = Depends(get_current_user)) -> UserRead:
    return UserRead.model_validate(current_user)
