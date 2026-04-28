from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.agent_run import AgentRun
from app.db.models.user import User
from app.schemas.agent_runs import AgentRunCreate
from app.services.tool_logs import create_tool_log


async def create_agent_run(
    session: AsyncSession,
    current_user: User,
    payload: AgentRunCreate,
) -> AgentRun:
    agent_run = AgentRun(
        user_id=current_user.id,
        prompt=payload.prompt.strip(),
        response=f"Placeholder response for prompt: {payload.prompt.strip()}",
        status="completed",
    )
    session.add(agent_run)
    await session.commit()
    await session.refresh(agent_run)
    await create_tool_log(
        session,
        agent_run,
        tool_name="travel_style_classifier",
        input_payload=payload.prompt.strip(),
        output_payload="Placeholder tool output for initial persistence wiring.",
        status="completed",
    )
    await session.refresh(agent_run, attribute_names=["tool_logs"])
    return agent_run
