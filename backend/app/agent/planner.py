from dataclasses import dataclass

from app.agent.graph import build_trip_planner_graph
from app.agent.tools.base import ToolContext
from app.agent.tools.registry import ToolRegistry
from app.schemas.agent_runs import AgentRunCreate


@dataclass(slots=True)
class ToolExecutionRecord:
    tool_name: str
    input_payload: str
    output_payload: str
    status: str


@dataclass(slots=True)
class PlannerResult:
    status: str
    response: str
    tool_logs: list[ToolExecutionRecord]


async def run_trip_planner(
    payload: AgentRunCreate,
    *,
    tool_registry: ToolRegistry | None,
    tool_context: ToolContext | None,
) -> PlannerResult:
    graph = build_trip_planner_graph()
    final_state = await graph.ainvoke(
        {
            "prompt": payload.prompt,
            "travel_profile": payload.travel_profile,
            "destination_name": payload.destination_name,
            "location_query": payload.location_query,
            "location_country_code": payload.location_country_code,
            "retrieval_top_k": payload.retrieval_top_k,
            "tool_registry": tool_registry,
            "tool_context": tool_context,
        }
    )

    return PlannerResult(
        status=str(final_state["status"]),
        response=str(
            final_state.get("final_response") or "\n".join(final_state["response_sections"])
        ),
        tool_logs=[
            ToolExecutionRecord(
                tool_name=tool_log["tool_name"],
                input_payload=tool_log["input_payload"],
                output_payload=tool_log["output_payload"],
                status=tool_log["status"],
            )
            for tool_log in final_state["tool_logs"]
        ],
    )
