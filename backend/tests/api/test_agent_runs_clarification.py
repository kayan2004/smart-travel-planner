"""Priority coverage for the /agent-runs HTTP contract added by the
clarification loop: interrupt -> resume -> complete, and the 3-round cap
falling back to a completed run. Uses httpx.ASGITransport directly (same
reasoning as test_auth.py: the real lifespan builds live DB/HTTP
resources this test doesn't want) but manually populates
app.state.resources/app.state.settings, since this route (unlike auth)
reads them.
"""

import httpx
import pytest
import pytest_asyncio

from app.agent import graph as graph_module
from app.agent.tools.registry import ToolRegistry
from app.core.config import get_settings
from app.db.dependencies import get_db_session
from app.schemas.claude import ExtractedRequestFields, TravelProfile
from main import app


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


@pytest_asyncio.fixture(scope="function", loop_scope="session")
async def api_client(engine):
    from app.db.session import create_session_factory

    factory = create_session_factory(engine)

    async def override_get_db_session():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.state.settings = get_settings()
    async with httpx.AsyncClient() as http_client:
        app.state.resources = {"tool_registry": ToolRegistry(), "http_client": http_client}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
    app.dependency_overrides.clear()
    app.state.resources = {}


@pytest.mark.asyncio(loop_scope="session")
async def test_clarification_interrupt_resume_complete_round_trip(
    monkeypatch, api_client, auth_headers
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

    first = await api_client.post(
        "/agent-runs", json={"prompt": "A trip to Lisbon"}, headers=auth_headers
    )
    assert first.status_code == 200
    first_body = first.json()
    assert first_body["status"] == "needs_input"
    assert "region" in first_body["question"].lower()
    thread_id = first_body["thread_id"]

    second = await api_client.post(
        "/agent-runs",
        json={
            "prompt": "A trip to Lisbon",
            "thread_id": thread_id,
            "clarification_answer": "Somewhere in Europe, ideally Portugal",
        },
        headers=auth_headers,
    )
    assert second.status_code == 201
    second_body = second.json()
    assert second_body["status"] in {"completed", "partial"}
    assert second_body["id"]


@pytest.mark.asyncio(loop_scope="session")
async def test_clarification_max_turns_cap_falls_back_to_completed(
    monkeypatch, api_client, auth_headers
):
    monkeypatch.setattr(
        graph_module,
        "extract_request_fields",
        _sequential_extractor(
            ExtractedRequestFields(
                destination_name=None,
                location_query=None,
                location_country_code=None,
                travel_profile=_profile("Flexible"),
            )
        ),
    )
    monkeypatch.setattr(graph_module, "synthesize_trip_response", _fake_synthesize)

    response = await api_client.post(
        "/agent-runs", json={"prompt": "Take me somewhere"}, headers=auth_headers
    )
    assert response.status_code == 200
    thread_id = response.json()["thread_id"]

    for _ in range(2):
        response = await api_client.post(
            "/agent-runs",
            json={
                "prompt": "Take me somewhere",
                "thread_id": thread_id,
                "clarification_answer": "still not sure",
            },
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert response.json()["status"] == "needs_input"

    final = await api_client.post(
        "/agent-runs",
        json={
            "prompt": "Take me somewhere",
            "thread_id": thread_id,
            "clarification_answer": "still not sure",
        },
        headers=auth_headers,
    )
    assert final.status_code == 201
    final_body = final.json()
    assert final_body["status"] in {"completed", "partial"}
    cap_logs = [
        log
        for log in final_body["tool_logs"]
        if log["tool_name"] == "clarification_loop" and "cap" in log["output_payload"].lower()
    ]
    assert len(cap_logs) == 1
