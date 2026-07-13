"""Route-level coverage for POST /feedback - deliberately unauthenticated
by design (anonymous session_uuid), so the only meaningful abuse guard is
rate limiting, not auth. See app/services/test_feedback.py for the
service-layer submit_feedback() coverage this doesn't duplicate.
"""

import uuid

import pytest


@pytest.mark.asyncio(loop_scope="session")
async def test_feedback_is_rate_limited_per_ip(api_client):
    payload = {
        "recommendation_id": 999999,
        "session_uuid": str(uuid.uuid4()),
        "verdict": 1,
    }

    last_response = None
    for _ in range(31):
        last_response = await api_client.post("/feedback", json=payload)

    assert last_response.status_code == 429
