from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import get_current_user
from app.core.byok import BYOKOverride, BYOKValidationError, build_byok_settings
from app.core.rate_limit import agent_run_ip_rate_limiter, agent_run_user_rate_limiter
from app.db.dependencies import get_db_session
from app.db.models.agent_run import AgentRun
from app.db.models.user import User
from app.schemas.agent_runs import AgentRunCreate, AgentRunRead, AgentRunSummary
from app.schemas.tool_logs import ToolLogRead
from app.services.agent_runs import (
    count_server_key_runs_for_user,
    create_agent_run,
    get_agent_run_for_user,
    list_agent_runs_for_user,
    sum_server_key_cost_this_month,
)
from app.services.llm_providers import LLMAuthenticationError
from app.services.recommendation_persistence import get_recommendations_for_agent_run
from app.agent.tools.base import ToolContext

router = APIRouter(prefix="/agent-runs", tags=["agent-runs"])


async def _free_runs_remaining(session: AsyncSession, user_id: int, free_limit: int) -> int:
    used = await count_server_key_runs_for_user(session, user_id)
    return max(0, free_limit - used)


async def _build_agent_run_read(
    session: AsyncSession, agent_run: AgentRun, *, free_runs_remaining: int | None = None
) -> AgentRunRead:
    recommendations = await get_recommendations_for_agent_run(session, agent_run.id)
    return AgentRunRead(
        id=agent_run.id,
        user_id=agent_run.user_id,
        prompt=agent_run.prompt,
        response=agent_run.response,
        status=agent_run.status,
        created_at=agent_run.created_at,
        tool_logs=[ToolLogRead.model_validate(log) for log in agent_run.tool_logs],
        recommendations=recommendations,
        free_runs_remaining=free_runs_remaining,
    )


@router.post("", response_model=AgentRunRead, status_code=status.HTTP_201_CREATED)
async def create_agent_run_route(
    payload: AgentRunCreate,
    request: Request,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    x_llm_api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
) -> AgentRunRead:
    client_ip = request.client.host if request.client else "unknown"
    if not await agent_run_ip_rate_limiter.check(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests from this address - please slow down.")
    if not await agent_run_user_rate_limiter.check(current_user.id):
        raise HTTPException(status_code=429, detail="Too many requests - please slow down.")

    tool_registry = request.app.state.resources.get("tool_registry")
    http_client = request.app.state.resources.get("http_client")
    settings = request.app.state.settings

    is_byok = bool(x_llm_api_key)

    # Free-tier gates - only for runs on the SERVER's key. A BYOK request is
    # the user's own spend, so it bypasses both. 402 (Payment Required) with
    # a machine-readable `reason` so the frontend can reliably branch to
    # "reveal the BYOK panel" instead of string-matching the message.
    if not is_byok:
        used = await count_server_key_runs_for_user(session, current_user.id)
        if used >= settings.free_server_runs_per_account:
            raise HTTPException(
                status_code=402,
                detail={
                    "reason": "free_quota_exhausted",
                    "message": (
                        "You've used your free trip plan. Add your own API key "
                        "below to keep planning."
                    ),
                },
            )
        spent = await sum_server_key_cost_this_month(session)
        if spent >= settings.server_key_monthly_budget_usd:
            raise HTTPException(
                status_code=402,
                detail={
                    "reason": "global_budget_exhausted",
                    "message": (
                        "The shared free tier is used up for this month. Add "
                        "your own API key below to keep planning."
                    ),
                },
            )

    if x_llm_api_key:
        if not payload.llm_provider or not payload.llm_model:
            raise HTTPException(
                status_code=400,
                detail="llm_provider and llm_model are required when X-LLM-API-Key is set.",
            )
        try:
            trip_settings = build_byok_settings(
                settings,
                BYOKOverride(
                    provider=payload.llm_provider,
                    model=payload.llm_model,
                    api_key=x_llm_api_key,
                ),
            )
        except BYOKValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        trip_settings = settings

    tool_context = ToolContext(
        settings=trip_settings,
        resources=request.app.state.resources,
        session=session,
        http_client=http_client,
        is_byok=is_byok,
    )

    try:
        agent_run = await create_agent_run(
            session,
            current_user,
            payload,
            tool_registry=tool_registry,
            tool_context=tool_context,
        )
    except LLMAuthenticationError as exc:
        raise HTTPException(
            status_code=401,
            detail="The provided API key was rejected by the provider.",
        ) from exc

    remaining = await _free_runs_remaining(
        session, current_user.id, settings.free_server_runs_per_account
    )
    return await _build_agent_run_read(session, agent_run, free_runs_remaining=remaining)


@router.get("", response_model=list[AgentRunSummary], status_code=status.HTTP_200_OK)
async def list_agent_runs_route(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[AgentRunSummary]:
    agent_runs = await list_agent_runs_for_user(
        session, current_user.id, limit=limit, offset=offset
    )
    return [AgentRunSummary.model_validate(agent_run) for agent_run in agent_runs]


@router.get("/{agent_run_id}", response_model=AgentRunRead, status_code=status.HTTP_200_OK)
async def get_agent_run_route(
    agent_run_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> AgentRunRead:
    agent_run = await get_agent_run_for_user(session, current_user.id, agent_run_id)
    if agent_run is None:
        raise HTTPException(status_code=404, detail="Trip plan not found.")
    remaining = await _free_runs_remaining(
        session, current_user.id, request.app.state.settings.free_server_runs_per_account
    )
    return await _build_agent_run_read(session, agent_run, free_runs_remaining=remaining)
