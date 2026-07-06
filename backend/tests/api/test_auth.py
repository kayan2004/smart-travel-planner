"""Priority 4 coverage: signup/login/me. Uses httpx.ASGITransport directly
(not TestClient) so the app's lifespan (loads the ML classifier model,
builds the tool registry) never runs - irrelevant here, and would require
artifacts/ml/best_model.joblib to exist just to test auth.
"""

import httpx
import pytest
import pytest_asyncio

from app.db.dependencies import get_db_session
from main import app


@pytest_asyncio.fixture(scope="function", loop_scope="session")
async def api_client(engine):
    from app.db.session import create_session_factory

    factory = create_session_factory(engine)

    async def override_get_db_session():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


@pytest.mark.asyncio(loop_scope="session")
async def test_signup_creates_user(api_client):
    response = await api_client.post(
        "/auth/signup", json={"email": "new-user@test.com", "password": "password123"}
    )
    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "new-user@test.com"
    assert body["is_active"] is True


@pytest.mark.asyncio(loop_scope="session")
async def test_signup_rejects_duplicate_email(api_client):
    payload = {"email": "dupe@test.com", "password": "password123"}
    first = await api_client.post("/auth/signup", json=payload)
    assert first.status_code == 201

    second = await api_client.post("/auth/signup", json=payload)
    assert second.status_code == 409


@pytest.mark.asyncio(loop_scope="session")
async def test_login_succeeds_with_correct_credentials(api_client):
    await api_client.post(
        "/auth/signup", json={"email": "login-ok@test.com", "password": "password123"}
    )

    response = await api_client.post(
        "/auth/login", json={"email": "login-ok@test.com", "password": "password123"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]


@pytest.mark.asyncio(loop_scope="session")
async def test_login_rejects_bad_password(api_client):
    await api_client.post(
        "/auth/signup", json={"email": "login-bad@test.com", "password": "password123"}
    )

    response = await api_client.post(
        "/auth/login", json={"email": "login-bad@test.com", "password": "wrong-password"}
    )
    assert response.status_code == 401


@pytest.mark.asyncio(loop_scope="session")
async def test_me_returns_current_user_with_valid_token(api_client, test_user, auth_headers):
    # Uses the auth_headers fixture (minted directly, not via a login round
    # trip) so this test isolates "/auth/me validates a token and returns
    # the right user" from "/auth/login issues a correct token" - the latter
    # is already covered by test_login_succeeds_with_correct_credentials.
    response = await api_client.get("/auth/me", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["email"] == test_user.email


@pytest.mark.asyncio(loop_scope="session")
async def test_me_rejects_missing_token(api_client):
    response = await api_client.get("/auth/me")
    assert response.status_code == 401
