"""Priority 6 coverage: a failing tool degrades a node to
tool_logs status=failed + graph status=partial, without raising -
app/agent/graph.py's core "tool failures are data, not exceptions" pattern
(see CLAUDE.md's "Conventions to follow when editing").

Also covers the BYOK auth-failure carve-out: an LLMAuthenticationError must
propagate (not degrade) for a BYOK request, but keep degrading gracefully
for the server's own default key - see graph.py's
extract_request_fields_node/synthesize_response_node.
"""

from unittest.mock import AsyncMock, patch

import pytest
from langgraph.runtime import Runtime
from pydantic import BaseModel

from app.agent.graph import (
    TripPlannerRuntime,
    extract_request_fields_node,
    retrieve_context_node,
)
from app.agent.tools.base import BaseTool, ToolContext
from app.agent.tools.registry import ToolRegistry
from app.core.config import get_settings
from app.services.llm_providers import LLMAuthenticationError


class _AlwaysFailsTool(BaseTool):
    name = "destination_context_retriever"
    description = "Test double that always raises."
    input_model = BaseModel

    async def arun(self, payload: BaseModel, context: ToolContext) -> BaseModel:
        raise RuntimeError("simulated RAG retrieval failure")


@pytest.mark.asyncio(loop_scope="session")
async def test_tool_failure_produces_failed_tool_log_and_partial_status(caplog):
    registry = ToolRegistry()
    registry.register(_AlwaysFailsTool())
    context = ToolContext(settings=get_settings(), resources={}, session=None, http_client=None)
    runtime = Runtime(context=TripPlannerRuntime(tool_registry=registry, tool_context=context))

    state = {
        "prompt": "a trip to nowhere",
        "retrieval_top_k": 5,
        "status": "completed",
        "response_sections": [],
        "tool_logs": [],
    }

    with caplog.at_level("ERROR", logger="app.agent.graph"):
        result = await retrieve_context_node(state, runtime)

    assert result["status"] == "partial"
    failed_logs = [log for log in result["tool_logs"] if log["status"] == "failed"]
    assert len(failed_logs) == 1
    assert failed_logs[0]["tool_name"] == "destination_context_retriever"
    # The exception's own message is deliberately NOT included in the API
    # response - only the type name is (app/agent/graph.py sanitizes this
    # before it flows into the response / frontend "Tool trail").
    assert "RuntimeError" in failed_logs[0]["output_payload"]
    assert "simulated RAG retrieval failure" not in failed_logs[0]["output_payload"]
    # ...but it's not lost - the full detail still reaches the server log
    # via logger.exception(), for operators to actually debug with.
    # record.getMessage() only returns the static "RAG retrieval failed"
    # string; the exception itself (with its message) lives on
    # record.exc_info, which is what logger.exception() attaches.
    assert any(
        record.exc_info is not None
        and "simulated RAG retrieval failure" in str(record.exc_info[1])
        for record in caplog.records
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_missing_tool_runtime_also_degrades_gracefully():
    """tool_registry=None (shared services unavailable) is the other real
    "can't run this tool" path this node handles - also data, not an
    exception.
    """
    runtime = Runtime(context=TripPlannerRuntime(tool_registry=None, tool_context=None))
    state = {
        "prompt": "a trip to nowhere",
        "retrieval_top_k": 5,
        "status": "completed",
        "response_sections": [],
        "tool_logs": [],
    }

    result = await retrieve_context_node(state, runtime)

    assert result["status"] == "partial"
    assert result["tool_logs"][0]["status"] == "failed"


@pytest.mark.asyncio(loop_scope="session")
async def test_byok_auth_failure_propagates_instead_of_degrading():
    context = ToolContext(
        settings=get_settings(),
        resources={},
        session=None,
        http_client=AsyncMock(),
        is_byok=True,
    )
    runtime = Runtime(context=TripPlannerRuntime(tool_registry=None, tool_context=context))
    state = {
        "prompt": "a trip to nowhere",
        "status": "completed",
        "response_sections": [],
        "tool_logs": [],
    }

    with patch(
        "app.agent.graph.extract_request_fields",
        new_callable=AsyncMock,
        side_effect=LLMAuthenticationError(status_code=401, context="test", body="rejected"),
    ):
        with pytest.raises(LLMAuthenticationError):
            await extract_request_fields_node(state, runtime)


@pytest.mark.asyncio(loop_scope="session")
async def test_server_key_auth_failure_still_degrades_gracefully():
    context = ToolContext(
        settings=get_settings(),
        resources={},
        session=None,
        http_client=AsyncMock(),
        is_byok=False,
    )
    runtime = Runtime(context=TripPlannerRuntime(tool_registry=None, tool_context=context))
    state = {
        "prompt": "a trip to nowhere",
        "status": "completed",
        "response_sections": [],
        "tool_logs": [],
    }

    with patch(
        "app.agent.graph.extract_request_fields",
        new_callable=AsyncMock,
        side_effect=LLMAuthenticationError(status_code=401, context="test", body="rejected"),
    ):
        result = await extract_request_fields_node(state, runtime)

    assert result["status"] == "partial"
    assert result["tool_logs"][0]["status"] == "failed"
