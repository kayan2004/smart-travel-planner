"""Route-level coverage for POST /agent-runs, focused on the BYOK feature:
default-key path, BYOK key path, allowlist rejection, rejected-key
handling, rate limiting, cross-request isolation, and log hygiene.

Unlike tests/api/test_auth.py's api_client fixture (which deliberately never
runs the app's lifespan), this route reads request.app.state.settings and
request.app.state.resources["tool_registry"/"http_client"] directly - both
of which only get populated by lifespan() in the real app. So this file's
fixture sets them manually instead of running the full lifespan, mirroring
what app/core/lifespan.py does for the pieces this route actually touches.

The tool registry is left empty on purpose: recommend_destinations/
retrieve_context/live_conditions all gracefully degrade to a "partial"
tool_logs entry when a tool isn't registered (the same "tool failures are
data" pattern covered in tests/agent/test_graph_tool_failure.py) - these
tests care about the LLM call sites (extraction, synthesis), not the full
pipeline, so keeping the other tools out of scope keeps the mocked HTTP
transport focused on exactly the requests worth asserting on.
"""

import json

import httpx
import pytest
import pytest_asyncio

from app.agent.tools.registry import ToolRegistry
from app.core.config import get_settings
from app.db.dependencies import get_db_session
from main import app

# Rate limiters are reset by conftest.py's autouse _reset_rate_limiters
# fixture before/after every test in this session, not just this file.


def _llm_mock_transport() -> httpx.MockTransport:
    """Answers both Anthropic (/v1/messages) and OpenAI (/v1/chat/completions)
    shaped requests with a fixed, valid response, and records every request
    for later assertions. `{}` is a deliberately valid-but-empty JSON body
    for the extraction call (ExtractedRequestFields' coercion falls back to
    an inferred travel_profile when the LLM doesn't return one - see
    app/services/llm.py's _coerce_extracted_fields) and a fine (if terse)
    "response" for the synthesis call.
    """
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        if request.url.path == "/v1/messages":
            return httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "{}"}],
                    "usage": {"input_tokens": 5, "output_tokens": 5},
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "{}"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
            },
        )

    transport = httpx.MockTransport(handler)
    transport.captured_requests = captured_requests  # type: ignore[attr-defined]
    return transport


def _rejecting_llm_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "invalid api key"}})

    return httpx.MockTransport(handler)


@pytest_asyncio.fixture(scope="function", loop_scope="session")
async def agent_runs_env(engine):
    """Sets app.state.settings/resources the way lifespan() would, but with
    a mocked http_client and an empty tool registry, then yields
    (api_client, mock_transport) for the test to drive.
    """
    from app.db.session import create_session_factory

    factory = create_session_factory(engine)

    async def override_get_db_session():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session

    base_settings = get_settings().model_copy(deep=True)
    base_settings.llm_provider = "anthropic"
    base_settings.anthropic.api_key = "server-default-key"
    base_settings.anthropic.model = "claude-haiku-4-5"
    app.state.settings = base_settings

    transport = _llm_mock_transport()
    mock_http_client = httpx.AsyncClient(transport=transport)
    app.state.resources = {
        "tool_registry": ToolRegistry(),
        "http_client": mock_http_client,
    }

    asgi_transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=asgi_transport, base_url="http://test") as client:
        yield client, transport

    await mock_http_client.aclose()
    app.dependency_overrides.clear()
    del app.state.settings
    del app.state.resources

    # This fixture builds its own session factory rather than depending on
    # the db_session fixture, so it never gets that fixture's truncate-
    # after-each-test cleanup - without this, every user/agent_run/etc a
    # test using this fixture creates persists in the real test Postgres
    # forever, across every future test run, not just within one session.
    # Confirmed live: an earlier debug run left rows that a later run's
    # list-ordering assertion then had to account for.
    from sqlalchemy import text

    from tests.conftest import TRUNCATE_TABLES

    async with engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE TABLE {TRUNCATE_TABLES} RESTART IDENTITY CASCADE"))


async def _signup_and_login(client: httpx.AsyncClient, email: str) -> str:
    await client.post("/auth/signup", json={"email": email, "password": "password123"})
    response = await client.post("/auth/login", json={"email": email, "password": "password123"})
    return response.json()["access_token"]


@pytest.mark.asyncio(loop_scope="session")
async def test_agent_run_uses_server_default_key_when_no_byok_header(agent_runs_env):
    client, transport = agent_runs_env
    token = await _signup_and_login(client, "byok-default@test.com")

    response = await client.post(
        "/agent-runs",
        json={"prompt": "A relaxing week in the mountains, medium budget."},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 201
    captured = transport.captured_requests
    assert captured, "expected at least one LLM call"
    for request in captured:
        assert request.headers["x-api-key"] == "server-default-key"


@pytest.mark.asyncio(loop_scope="session")
async def test_agent_run_uses_byok_key_when_header_present(agent_runs_env):
    client, transport = agent_runs_env
    token = await _signup_and_login(client, "byok-user@test.com")

    response = await client.post(
        "/agent-runs",
        json={
            "prompt": "A relaxing week in the mountains, medium budget.",
            "llm_provider": "openai",
            "llm_model": "gpt-5.4-nano",
        },
        headers={"Authorization": f"Bearer {token}", "X-LLM-API-Key": "user-byok-key"},
    )

    assert response.status_code == 201
    captured = transport.captured_requests
    openai_requests = [r for r in captured if r.url.path == "/v1/chat/completions"]
    assert openai_requests, "expected at least one OpenAI call"
    for request in openai_requests:
        assert request.headers["authorization"] == "Bearer user-byok-key"
    # The server's own default key must never be used for this request.
    assert all(r.headers.get("x-api-key") != "server-default-key" for r in captured)


@pytest.mark.asyncio(loop_scope="session")
async def test_agent_run_rejects_non_allowlisted_byok_combo(agent_runs_env):
    client, _transport = agent_runs_env
    token = await _signup_and_login(client, "byok-badmodel@test.com")

    response = await client.post(
        "/agent-runs",
        json={
            "prompt": "A relaxing week in the mountains, medium budget.",
            "llm_provider": "openai",
            "llm_model": "gpt-5.4-not-a-real-model",
        },
        headers={"Authorization": f"Bearer {token}", "X-LLM-API-Key": "user-byok-key"},
    )

    assert response.status_code == 400


@pytest.mark.asyncio(loop_scope="session")
async def test_agent_run_rejects_byok_header_without_provider_model_fields(agent_runs_env):
    client, _transport = agent_runs_env
    token = await _signup_and_login(client, "byok-missing-fields@test.com")

    response = await client.post(
        "/agent-runs",
        json={"prompt": "A relaxing week in the mountains, medium budget."},
        headers={"Authorization": f"Bearer {token}", "X-LLM-API-Key": "user-byok-key"},
    )

    assert response.status_code == 400


@pytest.mark.asyncio(loop_scope="session")
async def test_agent_run_returns_401_on_rejected_byok_key(agent_runs_env):
    client, _transport = agent_runs_env
    token = await _signup_and_login(client, "byok-rejected@test.com")

    app.state.resources["http_client"] = httpx.AsyncClient(transport=_rejecting_llm_transport())
    try:
        response = await client.post(
            "/agent-runs",
            json={
                "prompt": "A relaxing week in the mountains, medium budget.",
                "llm_provider": "openai",
                "llm_model": "gpt-5.4-nano",
            },
            headers={"Authorization": f"Bearer {token}", "X-LLM-API-Key": "bad-key"},
        )
    finally:
        await app.state.resources["http_client"].aclose()

    assert response.status_code == 401
    assert "bad-key" not in response.text


@pytest.mark.asyncio(loop_scope="session")
async def test_agent_run_server_key_failure_still_degrades_to_partial(agent_runs_env):
    client, _transport = agent_runs_env
    token = await _signup_and_login(client, "server-key-rejected@test.com")

    app.state.resources["http_client"] = httpx.AsyncClient(transport=_rejecting_llm_transport())
    try:
        response = await client.post(
            "/agent-runs",
            json={"prompt": "A relaxing week in the mountains, medium budget."},
            headers={"Authorization": f"Bearer {token}"},
        )
    finally:
        await app.state.resources["http_client"].aclose()

    assert response.status_code == 201
    assert response.json()["status"] == "partial"


@pytest.mark.asyncio(loop_scope="session")
async def test_agent_run_rate_limited_per_user(agent_runs_env):
    client, _transport = agent_runs_env
    token = await _signup_and_login(client, "rate-limited@test.com")
    headers = {"Authorization": f"Bearer {token}"}

    last_response = None
    for _ in range(11):
        last_response = await client.post(
            "/agent-runs",
            json={"prompt": "A relaxing week in the mountains, medium budget."},
            headers=headers,
        )

    assert last_response.status_code == 429


@pytest.mark.asyncio(loop_scope="session")
async def test_concurrent_byok_requests_do_not_cross_contaminate(agent_runs_env):
    client, transport = agent_runs_env
    token_a = await _signup_and_login(client, "concurrent-a@test.com")
    token_b = await _signup_and_login(client, "concurrent-b@test.com")

    import asyncio

    async def run(token: str, prompt_marker: str, api_key: str):
        return await client.post(
            "/agent-runs",
            json={
                "prompt": f"trip-request-{prompt_marker}: a relaxing week, medium budget.",
                "llm_provider": "openai",
                "llm_model": "gpt-5.4-nano",
            },
            headers={"Authorization": f"Bearer {token}", "X-LLM-API-Key": api_key},
        )

    response_a, response_b = await asyncio.gather(
        run(token_a, "A", "key-for-A"),
        run(token_b, "B", "key-for-B"),
    )

    assert response_a.status_code == 201
    assert response_b.status_code == 201

    captured = transport.captured_requests
    for request in captured:
        if request.url.path != "/v1/chat/completions":
            continue
        body = json.loads(request.content)
        prompt_text = json.dumps(body["messages"])
        if "trip-request-A" in prompt_text:
            assert request.headers["authorization"] == "Bearer key-for-A"
        elif "trip-request-B" in prompt_text:
            assert request.headers["authorization"] == "Bearer key-for-B"


@pytest.mark.asyncio(loop_scope="session")
async def test_byok_key_never_appears_in_logs(agent_runs_env, caplog):
    client, _transport = agent_runs_env
    token = await _signup_and_login(client, "byok-log-hygiene@test.com")

    secret_marker = "totally-secret-byok-key-xyz789"
    with caplog.at_level("INFO"):
        response = await client.post(
            "/agent-runs",
            json={
                "prompt": "A relaxing week in the mountains, medium budget.",
                "llm_provider": "openai",
                "llm_model": "gpt-5.4-nano",
            },
            headers={"Authorization": f"Bearer {token}", "X-LLM-API-Key": secret_marker},
        )

    assert response.status_code == 201
    for record in caplog.records:
        assert secret_marker not in record.getMessage()
    assert secret_marker not in response.text


@pytest.mark.asyncio(loop_scope="session")
async def test_list_agent_runs_returns_own_history_most_recent_first(agent_runs_env):
    client, _transport = agent_runs_env
    token = await _signup_and_login(client, "history-user@test.com")
    headers = {"Authorization": f"Bearer {token}"}

    first = await client.post(
        "/agent-runs", json={"prompt": "First trip: a weekend in the mountains."}, headers=headers
    )
    second = await client.post(
        "/agent-runs", json={"prompt": "Second trip: a week by the beach."}, headers=headers
    )
    assert first.status_code == 201
    assert second.status_code == 201

    list_response = await client.get("/agent-runs", headers=headers)
    assert list_response.status_code == 200
    body = list_response.json()
    assert [item["id"] for item in body] == [second.json()["id"], first.json()["id"]]
    # Summary shape only - no tool_logs/recommendations bloating the list.
    assert "tool_logs" not in body[0]
    assert "recommendations" not in body[0]


@pytest.mark.asyncio(loop_scope="session")
async def test_list_agent_runs_never_returns_another_users_runs(agent_runs_env):
    client, _transport = agent_runs_env
    owner_token = await _signup_and_login(client, "history-owner@test.com")
    await client.post(
        "/agent-runs",
        json={"prompt": "Owner's trip."},
        headers={"Authorization": f"Bearer {owner_token}"},
    )

    other_token = await _signup_and_login(client, "history-stranger@test.com")
    list_response = await client.get(
        "/agent-runs", headers={"Authorization": f"Bearer {other_token}"}
    )

    assert list_response.status_code == 200
    assert list_response.json() == []


@pytest.mark.asyncio(loop_scope="session")
async def test_get_agent_run_returns_full_detail_for_own_run(agent_runs_env):
    client, _transport = agent_runs_env
    token = await _signup_and_login(client, "history-detail@test.com")
    headers = {"Authorization": f"Bearer {token}"}

    created = await client.post(
        "/agent-runs", json={"prompt": "A detailed trip request."}, headers=headers
    )
    agent_run_id = created.json()["id"]

    detail_response = await client.get(f"/agent-runs/{agent_run_id}", headers=headers)

    assert detail_response.status_code == 200
    body = detail_response.json()
    assert body["id"] == agent_run_id
    assert body["prompt"] == "A detailed trip request."
    assert "tool_logs" in body
    assert "recommendations" in body


@pytest.mark.asyncio(loop_scope="session")
async def test_get_agent_run_404s_for_another_users_run_not_403(agent_runs_env):
    """404, not 403 - a user shouldn't be able to tell "exists but not
    yours" apart from "doesn't exist" by probing ids.
    """
    client, _transport = agent_runs_env
    owner_token = await _signup_and_login(client, "history-detail-owner@test.com")
    created = await client.post(
        "/agent-runs",
        json={"prompt": "Owner's private trip."},
        headers={"Authorization": f"Bearer {owner_token}"},
    )
    agent_run_id = created.json()["id"]

    other_token = await _signup_and_login(client, "history-detail-stranger@test.com")
    detail_response = await client.get(
        f"/agent-runs/{agent_run_id}", headers={"Authorization": f"Bearer {other_token}"}
    )

    assert detail_response.status_code == 404


@pytest.mark.asyncio(loop_scope="session")
async def test_get_agent_run_404s_for_a_nonexistent_id(agent_runs_env):
    client, _transport = agent_runs_env
    token = await _signup_and_login(client, "history-detail-missing@test.com")

    detail_response = await client.get(
        "/agent-runs/999999", headers={"Authorization": f"Bearer {token}"}
    )

    assert detail_response.status_code == 404


@pytest.mark.asyncio(loop_scope="session")
async def test_list_agent_runs_requires_auth(agent_runs_env):
    client, _transport = agent_runs_env
    response = await client.get("/agent-runs")
    assert response.status_code == 401
