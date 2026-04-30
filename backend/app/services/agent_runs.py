from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.planner import run_trip_planner
from app.agent.tools.base import ToolContext
from app.agent.tools.registry import ToolRegistry
from app.db.models.agent_run import AgentRun
from app.db.models.user import User
from app.schemas.agent_runs import AgentRunCreate
from app.services.discord_webhook import send_trip_plan_to_discord
from app.services.tool_logs import create_tool_log


async def create_agent_run(
    session: AsyncSession,
    current_user: User,
    payload: AgentRunCreate,
    *,
    tool_registry: ToolRegistry | None = None,
    tool_context: ToolContext | None = None,
) -> AgentRun:
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

    if tool_context is not None and tool_context.http_client is not None:
        try:
            await send_trip_plan_to_discord(
                tool_context.http_client,
                tool_context.settings,
                user_email=current_user.email,
                prompt=payload.prompt.strip(),
                response_text=planner_result.response,
                status=planner_result.status,
            )
            await create_tool_log(
                session,
                agent_run,
                tool_name="discord_webhook_delivery",
                input_payload=payload.prompt.strip(),
                output_payload="Trip plan delivered to Discord successfully.",
                status="completed",
            )
        except Exception as exc:
            await create_tool_log(
                session,
                agent_run,
                tool_name="discord_webhook_delivery",
                input_payload=payload.prompt.strip(),
                output_payload=(
                    f"Discord delivery failed: {type(exc).__name__}: {exc}"
                ),
                status="failed",
            )

    await session.refresh(agent_run, attribute_names=["tool_logs"])
    return agent_run
