"""Priority 4 coverage: signup/login/me. Uses httpx.ASGITransport directly
(not TestClient) so the app's lifespan (builds the tool registry, etc.)
never runs - irrelevant here. The api_client fixture itself now lives in
conftest.py since tests/api/test_feedback.py needs it too.
"""

import pytest


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


@pytest.mark.asyncio(loop_scope="session")
async def test_login_sets_httponly_access_token_cookie(api_client):
    await api_client.post(
        "/auth/signup", json={"email": "cookie-user@test.com", "password": "password123"}
    )
    response = await api_client.post(
        "/auth/login", json={"email": "cookie-user@test.com", "password": "password123"}
    )
    assert response.status_code == 200
    set_cookie_header = response.headers.get("set-cookie", "")
    assert "access_token=" in set_cookie_header
    assert "httponly" in set_cookie_header.lower()
    assert "samesite=lax" in set_cookie_header.lower()


@pytest.mark.asyncio(loop_scope="session")
async def test_me_authenticates_via_cookie_alone_no_authorization_header(api_client):
    await api_client.post(
        "/auth/signup", json={"email": "cookie-only@test.com", "password": "password123"}
    )
    await api_client.post(
        "/auth/login", json={"email": "cookie-only@test.com", "password": "password123"}
    )
    # httpx.AsyncClient persists Set-Cookie across requests on the same
    # client instance automatically - no Authorization header sent here.
    response = await api_client.get("/auth/me")
    assert response.status_code == 200
    assert response.json()["email"] == "cookie-only@test.com"


@pytest.mark.asyncio(loop_scope="session")
async def test_logout_clears_the_cookie(api_client):
    await api_client.post(
        "/auth/signup", json={"email": "logout-user@test.com", "password": "password123"}
    )
    await api_client.post(
        "/auth/login", json={"email": "logout-user@test.com", "password": "password123"}
    )
    logout_response = await api_client.post("/auth/logout")
    assert logout_response.status_code == 204

    me_response = await api_client.get("/auth/me")
    assert me_response.status_code == 401


@pytest.mark.asyncio(loop_scope="session")
async def test_logout_is_idempotent_without_an_existing_session(api_client):
    response = await api_client.post("/auth/logout")
    assert response.status_code == 204


@pytest.mark.asyncio(loop_scope="session")
async def test_login_is_rate_limited_per_ip(api_client):
    await api_client.post(
        "/auth/signup", json={"email": "rate-limited@test.com", "password": "password123"}
    )
    payload = {"email": "rate-limited@test.com", "password": "wrong-password"}

    last_response = None
    for _ in range(11):
        last_response = await api_client.post("/auth/login", json=payload)

    assert last_response.status_code == 429
