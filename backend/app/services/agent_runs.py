import logging
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agent.planner import run_trip_planner
from app.agent.tools.base import ToolContext
from app.agent.tools.registry import ToolRegistry
from app.db.models.agent_run import AgentRun
from app.db.models.user import User
from app.schemas.agent_runs import AgentRunCreate
from app.services.llm_providers.cost_tracking import reset_cost_accumulator
from app.services.recommendation_persistence import persist_recommendation_slate
from app.services.tool_logs import create_tool_log

logger = logging.getLogger(__name__)


async def create_agent_run(
    session: AsyncSession,
    current_user: User,
    payload: AgentRunCreate,
    *,
    tool_registry: ToolRegistry | None = None,
    tool_context: ToolContext | None = None,
) -> AgentRun:
    # Install a fresh per-run cost accumulator before any LLM call fires;
    # every provider call adds its estimated cost to this exact object (see
    # llm_providers/cost_tracking.py). Read after the graph finishes.
    cost_accumulator = reset_cost_accumulator()

    planner_result = await run_trip_planner(
        payload,
        tool_registry=tool_registry,
        tool_context=tool_context,
    )

    agent_run = AgentRun(
        user_id=current_user.id,
        prompt=payload.prompt.strip(),
        response=planner_result.response,
        status=planner_result.status,
        used_byok=bool(tool_context is not None and tool_context.is_byok),
        estimated_cost_usd=cost_accumulator.total_usd,
    )
    session.add(agent_run)
    await session.commit()
    await session.refresh(agent_run)

    for tool_log in planner_result.tool_logs:
        await create_tool_log(
            session,
            agent_run,
            tool_name=tool_log.tool_name,
            input_payload=tool_log.input_payload,
            output_payload=tool_log.output_payload,
            status=tool_log.status,
        )

    try:
        await persist_recommendation_slate(
            session, agent_run.id, planner_result.recommended_destinations
        )
        await create_tool_log(
            session,
            agent_run,
            tool_name="recommendation_persistence",
            input_payload=f"{len(planner_result.recommended_destinations)} slate row(s)",
            output_payload="Recommendation slate persisted successfully.",
            status="completed",
        )
    except Exception as exc:
        logger.exception("Recommendation slate persistence failed")
        await session.rollback()
        await create_tool_log(
            session,
            agent_run,
            tool_name="recommendation_persistence",
            input_payload=f"{len(planner_result.recommended_destinations)} slate row(s)",
            # Type only, not str(exc) - a DB error's full message can carry
            # internal details (constraint names, partial query text); the
            # full traceback goes to the server log above instead. This
            # payload flows straight into the API response and the
            # frontend's visible "Tool trail".
            output_payload=f"Recommendation slate persistence failed: {type(exc).__name__}.",
            status="failed",
        )

    await session.refresh(agent_run, attribute_names=["tool_logs", "recommendations"])
    return agent_run


async def list_agent_runs_for_user(
    session: AsyncSession,
    user_id: int,
    *,
    limit: int = 20,
    offset: int = 0,
) -> list[AgentRun]:
    statement = (
        select(AgentRun)
        .where(AgentRun.user_id == user_id, AgentRun.deleted_at.is_(None))
        # id as a tiebreaker - two rows can share created_at at whatever
        # timestamp precision the DB stores, and id order matches "most
        # recent first" anyway since it's autoincrement.
        .order_by(AgentRun.created_at.desc(), AgentRun.id.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await session.execute(statement)
    return list(result.scalars().all())


async def get_agent_run_for_user(
    session: AsyncSession,
    user_id: int,
    agent_run_id: int,
) -> AgentRun | None:
    """Scoped to user_id so one user can never fetch another's trip plan by
    guessing an id - returns None (route turns this into a 404, not a 403)
    rather than distinguishing "doesn't exist" from "not yours", which
    would leak which ids are in use.
    """
    statement = (
        select(AgentRun)
        .where(
            AgentRun.id == agent_run_id,
            AgentRun.user_id == user_id,
            AgentRun.deleted_at.is_(None),
        )
        .options(selectinload(AgentRun.tool_logs))
    )
    result = await session.execute(statement)
    return result.scalar_one_or_none()


async def count_server_key_runs_for_user(session: AsyncSession, user_id: int) -> int:
    """Lifetime count of this user's runs on the SERVER's key (not BYOK).

    Deliberately ignores deleted_at: a soft-deleted run still consumed a free
    run and cost real money, so counting it prevents a delete-then-rerun
    refund loophole on the per-account free tier.
    """
    statement = (
        select(func.count())
        .select_from(AgentRun)
        .where(AgentRun.user_id == user_id, AgentRun.used_byok.is_(False))
    )
    result = await session.execute(statement)
    return int(result.scalar_one())


async def sum_server_key_cost_this_month(session: AsyncSession) -> float:
    """Summed estimated cost of ALL users' server-key runs in the current UTC
    calendar month - the denominator of the global monthly budget gate.
    coalesce so an all-NULL/empty result is 0.0, not None.
    """
    now = datetime.now(timezone.utc)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    statement = select(
        func.coalesce(func.sum(AgentRun.estimated_cost_usd), 0.0)
    ).where(
        AgentRun.used_byok.is_(False),
        AgentRun.created_at >= month_start,
    )
    result = await session.execute(statement)
    return float(result.scalar_one() or 0.0)
