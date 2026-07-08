import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.agent_run import AgentRun
from app.db.models.tool_log import ToolLog

logger = logging.getLogger("app.tool_execution")


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

    # This is the one place every tool execution in the trip-planner pipeline
    # passes through - graph nodes, recommendation persistence, Discord
    # delivery (see app/services/agent_runs.py's call sites) - so logging
    # here gives pipeline-wide tracing without touching graph.py's node
    # functions individually.
    logger.info(
        "tool_execution",
        extra={
            "agent_run_id": agent_run.id,
            "tool_name": tool_name,
            "status": status,
            "input_payload_chars": len(input_payload),
            "output_payload_chars": len(output_payload),
        },
    )
    return tool_log
