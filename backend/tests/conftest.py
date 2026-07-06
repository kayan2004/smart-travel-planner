"""Shared pytest fixtures: an isolated test database (never the dev DB),
truncate-based test isolation, and full ORM model registration.

Test DB target is controlled by the DATABASE_URL env var - set it before
collecting tests (see the "Running Tests" section of backend/README.md).
Defaults to a local `smart_travel_assistant_test` database on the same
Postgres the dev stack already uses.
"""

import os
import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/smart_travel_assistant_test",
)

import asyncpg
import httpx
import numpy as np
import pytest_asyncio
from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import create_db_engine, create_session_factory

# Registers every ORM model's mapper before any query touches Recommendation
# or AgentRun - their relationship() targets are resolved by class name
# lazily, on first query, and every model on app.db.base.Base needs to have
# been imported somewhere first for that lookup to succeed (same pattern
# alembic/env.py uses for autogenerate, and scripts/train_ranker.py uses for
# the same reason).
from app.db.models import agent_run  # noqa: F401
from app.db.models import destination_document  # noqa: F401
from app.db.models import feedback  # noqa: F401
from app.db.models import recommendation  # noqa: F401
from app.db.models import tag_definition  # noqa: F401
from app.db.models import tool_log  # noqa: F401
from app.db.models import user  # noqa: F401

TRUNCATE_TABLES = (
    "feedback, recommendations, tool_logs, agent_runs, users, "
    "destination_documents, tag_definitions, destinations"
)


async def _ensure_test_database_exists(database_url: str) -> None:
    """Creates the test database if it doesn't exist yet.

    Runs a raw asyncpg connection to the `postgres` maintenance database
    (CREATE DATABASE cannot run inside a transaction block, which is why
    this doesn't go through the app's SQLAlchemy engine). This is awaited
    directly by the (already-async) _test_database_ready fixture below -
    it must NOT wrap itself in asyncio.run(), since that raises
    "asyncio.run() cannot be called from a running event loop" when called
    from inside a fixture that's already executing inside pytest-asyncio's
    event loop.
    """
    # asyncpg's DSN doesn't use the "+asyncpg" SQLAlchemy driver suffix.
    dsn = database_url.replace("postgresql+asyncpg://", "postgresql://")
    target_db = dsn.rsplit("/", 1)[1]
    maintenance_dsn = dsn.rsplit("/", 1)[0] + "/postgres"

    conn = await asyncpg.connect(maintenance_dsn)
    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", target_db
        )
        if not exists:
            await conn.execute(f'CREATE DATABASE "{target_db}"')
    finally:
        await conn.close()


def _run_migrations(database_url: str) -> None:
    # alembic/env.py is async-native for this project (its run_migrations_online()
    # calls asyncio.run(run_async_migrations()) internally) - invoking
    # alembic.command.upgrade() in-process from here would nest a second
    # asyncio.run() inside the event loop this (already-async) fixture is
    # running in, raising "asyncio.run() cannot be called from a running
    # event loop" (hit this exact error during execution). Shelling out as a
    # subprocess sidesteps the nesting entirely - alembic gets its own event
    # loop in its own process, same as running `alembic upgrade head` by hand.
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=BACKEND_DIR,
        env={**os.environ, "DATABASE_URL": database_url},
        check=True,
    )


@pytest_asyncio.fixture(scope="session", loop_scope="session", autouse=True)
async def _test_database_ready():
    """Session-scoped, autouse: creates + migrates the test DB exactly once."""
    settings = get_settings()
    assert "test" in settings.database_url, (
        f"Refusing to run tests against a non-test database: {settings.database_url}"
    )
    await _ensure_test_database_exists(settings.database_url)
    _run_migrations(settings.database_url)
    yield


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def engine(_test_database_ready):
    settings = get_settings()
    eng = create_db_engine(settings)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture(scope="function", loop_scope="session")
async def db_session(engine):
    factory = create_session_factory(engine)
    async with factory() as session:
        yield session
    # Truncate-after, not begin/rollback: several services under test
    # (persist_recommendation_slate, submit_feedback) commit internally,
    # which ends any outer transaction a rollback-based recipe would have
    # relied on. RESTART IDENTITY keeps primary keys predictable across
    # tests; CASCADE handles the recommendations -> feedback FK.
    async with engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE TABLE {TRUNCATE_TABLES} RESTART IDENTITY CASCADE"))


SEEDED_DESTINATIONS = [
    # (name, country, region, budget_level, tags)
    ("Aurora Bay", "Testland", "Europe", "low", {"0": 0.9, "1": 0.1}),
    ("Sunset Ridge", "Testland", "Europe", "medium", {"0": 0.2, "2": 0.8}),
    ("Marble Coast", "Testland", "Europe", "high", {"1": 0.7}),
    ("Copper Hollow", "Farland", "Asia", "low", {"2": 0.6, "3": 0.4}),
    ("Iron Vale", "Farland", "Asia", "medium", {}),
    ("Silver Delta", "Farland", "Asia", None, {"0": 0.5}),
    ("Golden Reach", "Otherland", "North America", "high", {"1": 0.9}),
    ("Quiet Hollow", "Otherland", "North America", "medium", {"3": 0.55}),
    ("Windmere", "Otherland", "North America", None, {}),
    ("East Fold", "Farland", "Asia", "high", {"2": 0.9, "0": 0.3}),
]


@pytest_asyncio.fixture(scope="function", loop_scope="session")
async def seeded_destinations(db_session):
    """10 destinations with real-shaped (unit-norm, 1024-dim) embeddings.

    Deterministic (seed=0) so cosine-order assertions in tests are stable
    across runs - each destination gets a distinct, reproducible direction
    in embedding space.
    """
    from app.db.models.destination import Destination

    rng = np.random.default_rng(0)
    destinations = []
    for name, country, region, budget_level, tags in SEEDED_DESTINATIONS:
        vector = rng.normal(0, 1, 1024)
        vector = (vector / np.linalg.norm(vector)).tolist()
        destination = Destination(
            name=name,
            country=country,
            region=region,
            budget_level=budget_level,
            details=f"{name} is a test fixture destination.",
            raw_sources={},
            source_provenance={},
            embedding=vector,
            embedding_model="test-fixture",
            embedding_version="v1",
            tags=tags,
        )
        db_session.add(destination)
        destinations.append(destination)
    await db_session.commit()
    for destination in destinations:
        await db_session.refresh(destination)
    return destinations


@pytest_asyncio.fixture(scope="function", loop_scope="session")
async def test_user(db_session):
    from app.core.security import hash_password
    from app.db.models.user import User

    user = User(
        email="fixture-user@test.com",
        hashed_password=hash_password("fixture-password-123"),
        full_name="Fixture User",
        is_active=True,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture(scope="function", loop_scope="session")
async def auth_headers(test_user):
    """A ready-to-use Authorization header for `test_user`, minted directly
    via create_access_token() rather than a real login round-trip - this is
    the fixture future endpoint tests (agent_runs, recommendations - both
    "behind JWT auth" per CLAUDE.md) should reach for when they need an
    authed request but aren't testing login itself.
    """
    from app.core.security import create_access_token

    token = create_access_token(test_user.email)
    return {"Authorization": f"Bearer {token}"}


def mock_voyage_transport(embedding: list[float] | None = None) -> httpx.MockTransport:
    """Builds an httpx.MockTransport that answers any Voyage /embeddings POST
    with a fixed, real-shaped embedding - no live Voyage call, ever.
    """
    if embedding is None:
        rng = np.random.default_rng(1)
        vector = rng.normal(0, 1, 1024)
        embedding = (vector / np.linalg.norm(vector)).tolist()

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        import json

        payload = json.loads(body)
        texts = payload["input"]
        return httpx.Response(
            200,
            json={"data": [{"embedding": embedding} for _ in texts]},
        )

    return httpx.MockTransport(handler)
