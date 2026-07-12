from fastapi import APIRouter, Depends, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.planner import PlannerNeedsInput
from app.agent.tools.base import ToolContext
from app.api.dependencies.auth import get_current_user
from app.db.dependencies import get_db_session
from app.db.models.user import User
from app.schemas.agent_runs import AgentRunCreate, AgentRunNeedsInput, AgentRunRead
from app.schemas.tool_logs import ToolLogRead
from app.services.agent_runs import create_agent_run
from app.services.recommendation_persistence import get_recommendations_for_agent_run

router = APIRouter(prefix="/agent-runs", tags=["agent-runs"])


@router.post("", response_model=AgentRunRead | AgentRunNeedsInput)
async def create_agent_run_route(
    payload: AgentRunCreate,
    request: Request,
    response: Response,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> AgentRunRead | AgentRunNeedsInput:
    tool_registry = request.app.state.resources.get("tool_registry")
    http_client = request.app.state.resources.get("http_client")
    tool_context = ToolContext(
        settings=request.app.state.settings,
        resources=request.app.state.resources,
        session=session,
        http_client=http_client,
    )
    result = await create_agent_run(
        session,
        current_user,
        payload,
        tool_registry=tool_registry,
        tool_context=tool_context,
    )

    if isinstance(result, PlannerNeedsInput):
        response.status_code = status.HTTP_200_OK
        return AgentRunNeedsInput(
            thread_id=result.thread_id, question=result.question, turn=result.turn
        )

    agent_run = result
    response.status_code = status.HTTP_201_CREATED
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
    )
