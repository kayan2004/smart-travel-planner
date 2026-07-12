"""Priority coverage: an interrupted run persists nothing (no AgentRun
row, no schema/migration change needed) until it actually completes -
see docs/superpowers/specs/2026-07-12-clarification-loop-design.md
Section 4."""

import httpx
import pytest
from sqlalchemy import select

from app.agent import graph as graph_module
from app.agent.planner import PlannerNeedsInput
from app.agent.tools.base import ToolContext
from app.agent.tools.registry import ToolRegistry
from app.core.config import get_settings
from app.db.models.agent_run import AgentRun
from app.schemas.agent_runs import AgentRunCreate
from app.schemas.claude import ExtractedRequestFields, TravelProfile
from app.services.agent_runs import create_agent_run


def _profile(region: str = "Flexible") -> TravelProfile:
    return TravelProfile(
        region=region,
        budget_level="medium",
        tourism_level="medium",
        has_hiking=False,
        has_beach=False,
        culture_score=5.0,
        luxury_score=5.0,
        family_friendly=5.0,
        nightlife_level=5.0,
        avg_temp_peak=20.0,
    )


def _sequential_extractor(*results: ExtractedRequestFields):
    calls = {"n": 0}

    async def _fake(http_client, settings, *, prompt):
        index = min(calls["n"], len(results) - 1)
        calls["n"] += 1
        return results[index]

    return _fake


async def _fake_synthesize(*args, **kwargs) -> str:
    return "synthesized response"


@pytest.mark.asyncio(loop_scope="session")
async def test_interrupted_run_persists_no_agent_run_row_until_completion(
    monkeypatch, db_session, test_user
):
    monkeypatch.setattr(
        graph_module,
        "extract_request_fields",
        _sequential_extractor(
            ExtractedRequestFields(
                destination_name="Lisbon",
                location_query="Lisbon, Portugal",
                location_country_code="PT",
                travel_profile=_profile("Flexible"),
            ),
            ExtractedRequestFields(
                destination_name="Lisbon",
                location_query="Lisbon, Portugal",
                location_country_code="PT",
                travel_profile=_profile("Europe"),
            ),
        ),
    )
    monkeypatch.setattr(graph_module, "synthesize_trip_response", _fake_synthesize)

    async with httpx.AsyncClient() as client:
        tool_context = ToolContext(
            settings=get_settings(), resources={}, session=db_session, http_client=client
        )
        registry = ToolRegistry()

        first = await create_agent_run(
            db_session,
            test_user,
            AgentRunCreate(prompt="A trip to Lisbon"),
            tool_registry=registry,
            tool_context=tool_context,
        )
        assert isinstance(first, PlannerNeedsInput)

        no_rows_yet = (await db_session.execute(select(AgentRun))).scalars().all()
        assert no_rows_yet == []

        second = await create_agent_run(
            db_session,
            test_user,
            AgentRunCreate(
                prompt="A trip to Lisbon",
                thread_id=first.thread_id,
                clarification_answer="Somewhere in Europe, ideally Portugal",
            ),
            tool_registry=registry,
            tool_context=tool_context,
        )

    assert isinstance(second, AgentRun)
    persisted_rows = (await db_session.execute(select(AgentRun))).scalars().all()
    assert len(persisted_rows) == 1
    assert persisted_rows[0].id == second.id
