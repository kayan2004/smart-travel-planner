from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.agent_run import AgentRun
from app.db.models.tool_log import ToolLog


async def create_tool_log(
    session: AsyncSession,
    agent_run: AgentRun,
    *,
    tool_name: str,
    input_payload: str,
    output_payload: str,
    status: str = "completed",
) -> ToolLog:
    tool_log = ToolLog(
        agent_run_id=agent_run.id,
        tool_name=tool_name,
        input_payload=input_payload,
        output_payload=output_payload,
        status=status,
    )
    session.add(tool_log)
    await session.commit()
    await session.refresh(tool_log)
    return tool_log
