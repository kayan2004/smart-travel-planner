# Pytest Suite + CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task, inline in this session on a feature branch (no subagent dispatch — this repo lives under a OneDrive-synced folder, and a prior subagent-driven run landed a commit in the wrong checkout because of it). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the first automated test suite (pytest + pytest-asyncio, async throughout) and
CI pipeline (GitHub Actions) for this backend, covering the six priority areas the spec names, plus
ruff/mypy gates — replacing the "no test pattern exists" gap documented in `CLAUDE.md`.

**Architecture:** Tests run against a **dedicated Postgres database** (`smart_travel_assistant_test`
locally, a GitHub Actions `services:` Postgres container in CI — not testcontainers, which adds
Docker-in-Docker complexity this project doesn't need since a real Postgres is already the norm
here), migrated with the real `alembic upgrade head` (never `create_all()` — this project has no
`create_all()` path left at all). Isolation is **truncate-all-tables after every test**, not
begin/rollback — several services under test (`persist_recommendation_slate`, `submit_feedback`)
commit internally, which silently defeats a rollback-based strategy (verified concretely during
planning: a rollback recipe passes when nothing commits and silently fails to isolate the moment a
service does). All external HTTP (Voyage, Gemini's REST fallback path, Anthropic, Open-Meteo,
Discord) is mocked via `httpx.MockTransport`; the `google-genai` SDK path is mocked via
`unittest.mock.patch` on the SDK client method directly, since that SDK does not go through the
shared `httpx.AsyncClient`. Endpoint tests use `httpx.AsyncClient(transport=httpx.ASGITransport(app=app))`
directly (not `TestClient`, not real app startup) with `app.dependency_overrides[get_db_session]`
swapped to the test session — `ASGITransport` never invokes the ASGI `lifespan` protocol at all, so
`load_travel_style_model()` and the rest of `app/core/lifespan.py` never run, which is exactly what
endpoint tests need since they're irrelevant to auth/feedback correctness.

**Tech Stack:** pytest 9.x, pytest-asyncio 1.x (session-scoped event loop + engine, function-scoped
session), pytest-cov, ruff, mypy, GitHub Actions with a `postgres` service container
(`pgvector/pgvector:0.8.2-pg17`, matching `docker-compose.yaml`'s dev image exactly).

## Global Constraints

- **Async fixtures throughout**, no sync test wrappers around async code.
- **No event-loop leakage between tests**: the async engine fixture and the async test loop must
  share the same scope (`loop_scope="session"` on both the fixture and every async test) — a
  session-scoped engine under a function-scoped loop raises `RuntimeError: Future attached to a
  different loop` the moment a second test runs. Verified concretely during planning against the
  real dev Postgres before writing this plan.
- **Isolation is truncate-based**, not transactional rollback — verified concretely during planning
  with a real committing service (`persist_recommendation_slate`): rollback-based isolation looks
  correct until the first service-level `commit()`, then silently leaks state into the next test.
- **Every ORM model module must be imported before the first query touches `Recommendation` or
  `AgentRun`** — both have `relationship(...)` targets resolved by class name lazily, and hit this
  in exactly this order during planning verification: `AgentRun` -> `User` -> `ToolLog`. Import all
  seven model modules (`agent_run`, `destination_document`, `feedback`, `recommendation`,
  `tag_definition`, `tool_log`, `user`) in `conftest.py`, the same pattern `alembic/env.py` already
  uses for autogenerate and `scripts/train_ranker.py` now uses too.
- **`DATABASE_URL` env var is the one mechanism** both local test runs and CI use to point
  `get_settings()` at the test database instead of the dev database — verified working (pydantic-settings
  prioritizes real process env vars over the `.env` file it also reads).
- **Never touch the dev database.** All fixtures target `smart_travel_assistant_test` (local) or the
  CI-only Postgres service container - never `smart_travel_assistant`.
- **Mock every external HTTP boundary** (Voyage, Anthropic, Open-Meteo, Discord) via
  `httpx.MockTransport`; mock the `google-genai` SDK client method directly (it doesn't use httpx).
  pgvector operations (`cosine_distance`, the HNSW index, `destinations` filters) run against the
  real test Postgres, never mocked.
- **Deterministic**: every RNG-driven fixture (synthetic embeddings) uses a fixed seed
  (`numpy.random.default_rng(0)`); no test depends on wall-clock time.
- **Coverage target ~70% of `app/services` + `app/agent`** — a target, not a hard gate to chase to
  100%; CI reports coverage but the pass/fail gate is "tests pass," not a coverage percentage
  threshold.
- **mypy is configured leniently on purpose**: this codebase has 39 pre-existing type errors (mostly
  in `app/agent/graph.py`'s LangGraph nodes, which return partial-state dicts annotated as the full
  `TripPlannerState` TypedDict — a real, pre-existing, LangGraph-idiomatic pattern, not a bug this
  PR should refactor). A scoped per-module override suppresses exactly those pre-existing errors so
  mypy is a meaningful gate against *new* type errors without turning "add tests" into "retrofit
  types onto the whole codebase."
- **This is the first test suite in this project** — `CLAUDE.md`'s "Known gaps" section currently
  says "no automated tests or CI... there's no existing pattern to extend." This plan's last task
  updates that line; every fixture/pattern this plan introduces is deliberately the thing future
  sessions should extend, not route around.

---

## File Structure

**Create:**
- `backend/tests/__init__.py` — empty, makes `tests` a package so imports resolve consistently.
- `backend/tests/conftest.py` — the whole fixture story: test DB creation + migration, session-scoped
  engine/loop, function-scoped session with truncate-after teardown, full model-module registration,
  seeded destination corpus, authed user + JWT, `httpx.MockTransport` factory helpers.
- `backend/tests/services/test_destination_recommendations.py` — priority 1.
- `backend/tests/services/test_feedback.py` — priority 2.
- `backend/tests/services/test_recommendation_persistence.py` — priority 3.
- `backend/tests/api/test_auth.py` — priority 4.
- `backend/tests/services/test_llm_providers.py` — priority 5.
- `backend/tests/agent/test_graph_tool_failure.py` — priority 6.
- `.github/workflows/ci.yml` — Postgres service, `uv sync`, `alembic upgrade head`, `pytest --cov`,
  `ruff check`, `mypy` as separate steps.

**Modify:**
- `backend/pyproject.toml` — `[tool.pytest.ini_options]`, `[tool.coverage.run]`, `[tool.ruff]`,
  `[tool.mypy]` + one `[[tool.mypy.overrides]]` block; fix the one real `ruff` finding
  (`estimate_text_tokens` unused import in `destination_ingestion.py`).
- `backend/README.md` — "Running Tests" section (local setup, coverage, CI badge).
- `CLAUDE.md` — "Known gaps" section: replace "no automated tests or CI" framing with a pointer to
  `backend/tests/conftest.py`'s fixtures as the pattern to extend.

---

### Task 0: Branch + commit already-verified dev dependencies

**Files:**
- Modify (already changed on disk during pre-planning verification, needs committing):
  `backend/pyproject.toml`, `backend/uv.lock`

- [ ] **Step 1: Create the feature branch**

```bash
cd "C:/Users/Kayan/OneDrive/Desktop/projects/sef_projects/smart_travel_assistant"
git status --short
git checkout -b feat/pytest-suite-and-ci
```

Expected: only `backend/pyproject.toml`/`backend/uv.lock` modified (pytest/pytest-asyncio/pytest-cov/ruff/mypy
added as dev deps during pre-planning verification), then confirmation you're on the new branch.

- [ ] **Step 2: Verify the dev toolchain still imports cleanly**

```bash
cd backend
uv run python -c "import pytest, pytest_asyncio, pytest_cov, ruff; print('OK')" 2>&1 | tail -5
uv run ruff --version
uv run mypy --version
```

Expected: no import errors (note: `ruff`/`mypy` are CLI tools, not always importable as Python
modules depending on how they're installed — if the `import ruff` line errors with `ModuleNotFoundError`,
that's fine and expected; the `uv run ruff --version`/`uv run mypy --version` lines are the real check).

- [ ] **Step 3: Commit**

```bash
cd ..
git add backend/pyproject.toml backend/uv.lock
git commit -m "chore(deps): add pytest/pytest-asyncio/pytest-cov/ruff/mypy as dev dependencies"
```

---

### Task 1: Test database fixtures — engine, session, isolation

**Files:**
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/conftest.py`
- Modify: `backend/pyproject.toml` (`[tool.pytest.ini_options]`)

**Interfaces:**
- Produces: `engine` fixture (session-scoped, `loop_scope="session"`) — an `AsyncEngine` pointed at
  `smart_travel_assistant_test`. `db_session` fixture (function-scoped, `loop_scope="session"`) — an
  `AsyncSession` from that engine, truncating every app table after each test. Every later task's
  tests depend on `db_session`.

- [ ] **Step 1: Create the package marker**

Create `backend/tests/__init__.py` as a completely empty file (zero bytes — just makes `tests` a
proper Python package so `from tests.conftest import ...` imports work consistently across test
modules).

- [ ] **Step 2: Add pytest config to `pyproject.toml`**

Add this new section to `backend/pyproject.toml` (anywhere after `[dependency-groups]`):

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "session"
asyncio_default_test_loop_scope = "session"
testpaths = ["tests"]

[tool.coverage.run]
source = ["app"]
omit = ["app/agent/tools/*/__pycache__/*"]
```

`asyncio_default_test_loop_scope = "session"` matters as much as the fixture-loop-scope setting: a
session-scoped `engine`/`db_session` fixture under a *function*-scoped test loop (pytest-asyncio's
own default) raises `RuntimeError: Future attached to a different loop` the moment a second test
runs - confirmed by deliberately hitting this during pre-planning verification. Every test in this
plan still adds `@pytest.mark.asyncio(loop_scope="session")` explicitly, which is redundant once
this config is set - kept anyway so the pattern stays copy-pasteable even if a future session
changes this config without noticing the coupling.

- [ ] **Step 3: Create `backend/tests/conftest.py`**

```python
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
    this doesn't go through the app's SQLAlchemy engine). Awaited directly
    by the (already-async) _test_database_ready fixture below - it must NOT
    wrap itself in asyncio.run(), since that raises "asyncio.run() cannot
    be called from a running event loop" when called from inside a fixture
    that's already executing inside pytest-asyncio's event loop (hit this
    exact error during execution, fixed by removing the asyncio.run() wrapper).
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
```

- [ ] **Step 4: Create the test database and verify the fixture works standalone**

```bash
cd backend
docker exec smart_travel_assistant-db-1 psql -U postgres -c "SELECT 1 FROM pg_database WHERE datname='smart_travel_assistant_test'" | grep -q 1 || docker exec smart_travel_assistant-db-1 psql -U postgres -c "CREATE DATABASE smart_travel_assistant_test"
cat > tests/test_conftest_smoke.py << 'EOF'
import pytest
from sqlalchemy import text


@pytest.mark.asyncio(loop_scope="session")
async def test_db_session_fixture_works(db_session):
    result = await db_session.execute(text("SELECT 1"))
    assert result.scalar() == 1
EOF
uv run pytest tests/test_conftest_smoke.py -v
rm tests/test_conftest_smoke.py
```

Expected: `1 passed`. This confirms the autouse fixture created + migrated the test DB and the
session fixture connects successfully. The smoke test file is deleted immediately after — it existed
only to prove the fixture works before other tests depend on it.

- [ ] **Step 5: Commit**

```bash
cd ..
git add backend/tests/__init__.py backend/tests/conftest.py backend/pyproject.toml
git commit -m "test: add async test-DB fixtures with truncate-based isolation"
```

---

### Task 2: Seed fixtures — destinations corpus, authed user, mock transports

**Files:**
- Modify: `backend/tests/conftest.py`

**Interfaces:**
- Produces: `seeded_destinations` fixture (function-scoped) — inserts 10 `Destination` rows with
  real-shaped (unit-norm, 1024-dim, seeded RNG) embeddings, varied `budget_level`/`region`/`tags`,
  returns the list of inserted `Destination` ORM objects. `test_user` fixture — a committed `User`
  row (consumed by Tasks 4/5's persistence tests, which need a real `agent_runs.user_id`).
  `auth_headers` fixture — a ready `{"Authorization": "Bearer ..."}` header for `test_user` (consumed
  by Task 6's `/auth/me` test). `mock_voyage_transport(embedding=None)` factory function — builds an
  `httpx.MockTransport` returning a fixed embedding for any `/embeddings` POST.

- [ ] **Step 1: Add the destination-seeding fixture**

Add to `backend/tests/conftest.py` (after the model-registration imports, before `TRUNCATE_TABLES`):

```python
import numpy as np
```

Add after the `db_session` fixture:

```python
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


def mock_voyage_transport(embedding: list[float] | None = None) -> "httpx.MockTransport":
    """Builds an httpx.MockTransport that answers any Voyage /embeddings POST
    with a fixed, real-shaped embedding - no live Voyage call, ever.
    """
    import httpx

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
```

Add `import httpx` to the top-level imports of `conftest.py` (needed for the type annotation above
and reused by later tasks):

```python
import httpx
```

- [ ] **Step 2: Verify the seed fixtures work standalone**

```bash
cd backend
cat > tests/test_conftest_smoke.py << 'EOF'
import pytest


@pytest.mark.asyncio(loop_scope="session")
async def test_seeded_destinations_fixture(seeded_destinations):
    assert len(seeded_destinations) == 10
    assert all(len(d.embedding) == 1024 for d in seeded_destinations)


@pytest.mark.asyncio(loop_scope="session")
async def test_auth_headers_fixture(auth_headers):
    assert auth_headers["Authorization"].startswith("Bearer ")
EOF
uv run pytest tests/test_conftest_smoke.py -v
rm tests/test_conftest_smoke.py
```

Expected: `2 passed`.

- [ ] **Step 3: Commit**

```bash
cd ..
git add backend/tests/conftest.py
git commit -m "test: add seeded-destination, authed-user, and Voyage-mock fixtures"
```

---

### Task 3: Recommendation service tests (priority 1)

**Files:**
- Create: `backend/tests/services/__init__.py`
- Create: `backend/tests/services/test_destination_recommendations.py`

**Interfaces:**
- Consumes: `db_session`, `seeded_destinations` fixtures (Task 1/2); `mock_voyage_transport()`
  (Task 2); `recommend_destinations(session, http_client, settings, payload)` from
  `app.services.destination_recommendations` (existing).

- [ ] **Step 1: Create `backend/tests/services/__init__.py`** (empty file)

- [ ] **Step 2: Write the tests**

```python
"""Priority 1 coverage: app/services/destination_recommendations.py."""

import httpx
import pytest

from app.core.config import get_settings
from app.schemas.recommendations import DestinationRecommendationRequest
from app.services.destination_recommendations import recommend_destinations
from tests.conftest import mock_voyage_transport


@pytest.mark.asyncio(loop_scope="session")
async def test_budget_ceiling_allows_lower_and_equal_levels(db_session, seeded_destinations):
    settings = get_settings()
    transport = mock_voyage_transport()
    async with httpx.AsyncClient(transport=transport) as client:
        payload = DestinationRecommendationRequest(
            query_text="a medium-budget trip", budget_level="medium", limit=10, min_candidates=10
        )
        response = await recommend_destinations(db_session, client, settings, payload)

    returned_budgets = {item.budget_level for item in response.results}
    # "high" destinations must never appear when the ceiling is "medium".
    assert "high" not in returned_budgets


@pytest.mark.asyncio(loop_scope="session")
async def test_budget_none_passes_through_null_budget_destinations(db_session, seeded_destinations):
    """Destinations with budget_level=None always pass the filter, regardless
    of the requested ceiling - _fetch_ranked_candidates ORs in `is_(None)`.
    """
    settings = get_settings()
    transport = mock_voyage_transport()
    async with httpx.AsyncClient(transport=transport) as client:
        payload = DestinationRecommendationRequest(
            query_text="a low-budget trip", budget_level="low", limit=10, min_candidates=10
        )
        response = await recommend_destinations(db_session, client, settings, payload)

    returned_names = {item.destination for item in response.results}
    # "Silver Delta" and "Windmere" have budget_level=None in the seed fixture.
    assert "Silver Delta" in returned_names
    assert "Windmere" in returned_names


@pytest.mark.asyncio(loop_scope="session")
async def test_region_flexible_sentinel_skips_region_filter(db_session, seeded_destinations):
    settings = get_settings()
    transport = mock_voyage_transport()
    async with httpx.AsyncClient(transport=transport) as client:
        payload = DestinationRecommendationRequest(
            query_text="anywhere is fine", region="flexible", limit=10, min_candidates=10
        )
        response = await recommend_destinations(db_session, client, settings, payload)

    returned_regions = {item.region for item in response.results}
    # Multiple regions present in the seed corpus - "flexible" must not narrow to one.
    assert len(returned_regions) > 1


@pytest.mark.asyncio(loop_scope="session")
async def test_region_filter_narrows_to_requested_region(db_session, seeded_destinations):
    settings = get_settings()
    transport = mock_voyage_transport()
    async with httpx.AsyncClient(transport=transport) as client:
        payload = DestinationRecommendationRequest(
            query_text="somewhere in Asia", region="Asia", limit=10, min_candidates=10
        )
        response = await recommend_destinations(db_session, client, settings, payload)

    returned_regions = {item.region for item in response.results}
    assert returned_regions == {"Asia"}


@pytest.mark.asyncio(loop_scope="session")
async def test_results_are_ordered_by_cosine_similarity_descending(db_session, seeded_destinations):
    settings = get_settings()
    # Use the exact embedding of one seeded destination as the "user profile"
    # so its cosine similarity to itself is the maximum possible (1.0),
    # guaranteeing a predictable top result.
    target = seeded_destinations[0]
    transport = mock_voyage_transport(embedding=list(target.embedding))
    async with httpx.AsyncClient(transport=transport) as client:
        payload = DestinationRecommendationRequest(
            query_text="find me something like Aurora Bay", limit=10, min_candidates=10
        )
        response = await recommend_destinations(db_session, client, settings, payload)

    scores = [item.score for item in response.results]
    assert scores == sorted(scores, reverse=True)
    assert response.results[0].destination == target.name
    assert response.results[0].score == pytest.approx(1.0, abs=1e-3)


@pytest.mark.asyncio(loop_scope="session")
async def test_relaxes_constraints_when_too_few_candidates_survive(db_session, seeded_destinations):
    """required_tags with an impossibly high threshold should eliminate every
    candidate under strict filtering, forcing the relaxed-fallback path.
    """
    settings = get_settings()
    transport = mock_voyage_transport()
    async with httpx.AsyncClient(transport=transport) as client:
        payload = DestinationRecommendationRequest(
            query_text="something with a nonexistent tag",
            required_tags=["9"],
            tag_weight_threshold=0.99,
            limit=5,
            min_candidates=5,
        )
        response = await recommend_destinations(db_session, client, settings, payload)

    assert response.used_relaxed_constraints is True
    assert response.count > 0


@pytest.mark.asyncio(loop_scope="session")
async def test_feature_snapshot_matches_request_constraints(db_session, seeded_destinations):
    settings = get_settings()
    transport = mock_voyage_transport()
    async with httpx.AsyncClient(transport=transport) as client:
        payload = DestinationRecommendationRequest(
            query_text="a medium-budget trip to Asia",
            budget_level="medium",
            region="Asia",
            limit=10,
            min_candidates=10,
        )
        response = await recommend_destinations(db_session, client, settings, payload)

    for item in response.results:
        assert item.features.cosine_sim == item.score
        assert item.features.region_match is True
        if item.budget_level is not None:
            # budget_delta = BUDGET_ORDER[destination] - BUDGET_ORDER[requested]
            assert isinstance(item.features.budget_delta, int)
```

- [ ] **Step 3: Run the tests**

```bash
cd backend
uv run pytest tests/services/test_destination_recommendations.py -v
```

Expected: `7 passed`.

- [ ] **Step 4: Commit**

```bash
cd ..
git add backend/tests/services/__init__.py backend/tests/services/test_destination_recommendations.py
git commit -m "test: cover destination recommendation service (priority 1)"
```

---

### Task 4: Feedback service tests (priority 2)

**Files:**
- Create: `backend/tests/services/test_feedback.py`

**Interfaces:**
- Consumes: `db_session`, `seeded_destinations`, `test_user` fixtures. `submit_feedback(session,
  payload, *, channel="web")` and `RecommendationNotFoundError` from `app.services.feedback`
  (existing). `FeedbackCreate` from `app.schemas.feedback` (existing).

- [ ] **Step 1: Write the tests**

```python
"""Priority 2 coverage: app/services/feedback.py."""

import uuid

import pytest
from sqlalchemy import insert

from app.db.models.agent_run import AgentRun
from app.db.models.recommendation import Recommendation
from app.schemas.feedback import FeedbackCreate
from app.services.feedback import RecommendationNotFoundError, submit_feedback


async def _make_recommendation(db_session, seeded_destinations, test_user) -> int:
    agent_run = AgentRun(
        user_id=test_user.id, prompt="p", response="r", status="completed"
    )
    db_session.add(agent_run)
    await db_session.flush()
    result = await db_session.execute(
        insert(Recommendation)
        .values(
            agent_run_id=agent_run.id,
            destination_id=seeded_destinations[0].id,
            rank_position=1,
            score=0.9,
            features={"cosine_sim": 0.9, "tag_match_count": 0, "budget_delta": None, "region_match": True},
        )
        .returning(Recommendation.id)
    )
    await db_session.commit()
    return result.scalar_one()


@pytest.mark.asyncio(loop_scope="session")
async def test_submit_feedback_creates_row(db_session, seeded_destinations, test_user):
    recommendation_id = await _make_recommendation(db_session, seeded_destinations, test_user)
    session_uuid = uuid.uuid4()

    result = await submit_feedback(
        db_session,
        FeedbackCreate(recommendation_id=recommendation_id, session_uuid=session_uuid, verdict=1),
    )

    assert result.recommendation_id == recommendation_id
    assert result.session_uuid == session_uuid
    assert result.verdict == 1
    assert result.channel == "web"


@pytest.mark.asyncio(loop_scope="session")
async def test_submit_feedback_is_idempotent_on_recommendation_and_session(
    db_session, seeded_destinations, test_user
):
    recommendation_id = await _make_recommendation(db_session, seeded_destinations, test_user)
    session_uuid = uuid.uuid4()

    first = await submit_feedback(
        db_session,
        FeedbackCreate(recommendation_id=recommendation_id, session_uuid=session_uuid, verdict=1),
    )
    second = await submit_feedback(
        db_session,
        FeedbackCreate(recommendation_id=recommendation_id, session_uuid=session_uuid, verdict=-1),
    )

    # Same row updated in place, not a second row inserted.
    assert first.id == second.id
    assert second.verdict == -1


@pytest.mark.asyncio(loop_scope="session")
async def test_submit_feedback_raises_for_unknown_recommendation(db_session):
    with pytest.raises(RecommendationNotFoundError):
        await submit_feedback(
            db_session,
            FeedbackCreate(recommendation_id=999999, session_uuid=uuid.uuid4(), verdict=1),
        )
```

- [ ] **Step 2: Run the tests**

```bash
cd backend
uv run pytest tests/services/test_feedback.py -v
```

Expected: `3 passed`.

- [ ] **Step 3: Commit**

```bash
cd ..
git add backend/tests/services/test_feedback.py
git commit -m "test: cover feedback upsert idempotency (priority 2)"
```

---

### Task 5: Slate persistence tests (priority 3)

**Files:**
- Create: `backend/tests/services/test_recommendation_persistence.py`

**Interfaces:**
- Consumes: `db_session`, `seeded_destinations`, `test_user` fixtures.
  `persist_recommendation_slate(session, agent_run_id, recommended_destinations)` and
  `get_recommendations_for_agent_run(session, agent_run_id)` from
  `app.services.recommendation_persistence` (existing).

- [ ] **Step 1: Write the tests**

```python
"""Priority 3 coverage: app/services/recommendation_persistence.py.

This is specifically the cross-metadata-base Core-insert path - Destination
lives on its own DeclarativeBase (DestinationCorpusBase), so a plain
session.add()/session.add_all() for Recommendation triggers a
NoReferencedTableError at flush time (see backend/README.md's "Cross-metadata-base
ORM flush bug" section). persist_recommendation_slate works around this with
a Core insert(...).values([...]).returning(...) - these tests exist
specifically to catch a regression back to session.add().
"""

import pytest

from app.db.models.agent_run import AgentRun
from app.services.recommendation_persistence import (
    get_recommendations_for_agent_run,
    persist_recommendation_slate,
)


async def _make_agent_run(db_session, test_user) -> int:
    agent_run = AgentRun(user_id=test_user.id, prompt="p", response="r", status="completed")
    db_session.add(agent_run)
    await db_session.commit()
    await db_session.refresh(agent_run)
    return agent_run.id


@pytest.mark.asyncio(loop_scope="session")
async def test_persists_the_full_slate_not_just_top_result(db_session, seeded_destinations, test_user):
    agent_run_id = await _make_agent_run(db_session, test_user)
    slate = [
        {
            "destination_id": str(destination.id),
            "rank_position": index + 1,
            "score": round(1.0 - index * 0.1, 4),
            "features": {
                "cosine_sim": round(1.0 - index * 0.1, 4),
                "tag_match_count": 0,
                "budget_delta": None,
                "region_match": True,
            },
        }
        for index, destination in enumerate(seeded_destinations[:3])
    ]

    persisted = await persist_recommendation_slate(db_session, agent_run_id, slate)

    assert len(persisted) == 3
    assert [row.rank_position for row in persisted] == [1, 2, 3]


@pytest.mark.asyncio(loop_scope="session")
async def test_features_are_captured_verbatim(db_session, seeded_destinations, test_user):
    agent_run_id = await _make_agent_run(db_session, test_user)
    features = {"cosine_sim": 0.8123, "tag_match_count": 2, "budget_delta": -1, "region_match": False}
    slate = [
        {
            "destination_id": str(seeded_destinations[0].id),
            "rank_position": 1,
            "score": 0.8123,
            "features": features,
        }
    ]

    await persist_recommendation_slate(db_session, agent_run_id, slate)
    rows = await get_recommendations_for_agent_run(db_session, agent_run_id)

    assert rows[0].features == features


@pytest.mark.asyncio(loop_scope="session")
async def test_empty_slate_persists_nothing(db_session, test_user):
    agent_run_id = await _make_agent_run(db_session, test_user)
    persisted = await persist_recommendation_slate(db_session, agent_run_id, [])
    assert persisted == []
```

- [ ] **Step 2: Run the tests**

```bash
cd backend
uv run pytest tests/services/test_recommendation_persistence.py -v
```

Expected: `3 passed`.

- [ ] **Step 3: Commit**

```bash
cd ..
git add backend/tests/services/test_recommendation_persistence.py
git commit -m "test: cover slate persistence cross-metadata-base insert path (priority 3)"
```

---

### Task 6: Auth endpoint tests (priority 4)

**Files:**
- Create: `backend/tests/api/__init__.py`
- Create: `backend/tests/api/test_auth.py`

**Interfaces:**
- Consumes: `engine` fixture (Task 1). `main.app`, `app.db.dependencies.get_db_session` (existing).
- Uses `httpx.AsyncClient(transport=httpx.ASGITransport(app=app))` directly (verified during
  planning that this never invokes the ASGI `lifespan` protocol, so `app/core/lifespan.py`'s heavy
  startup - `load_travel_style_model()`, tool registry construction - never runs; irrelevant to auth).

- [ ] **Step 1: Create `backend/tests/api/__init__.py`** (empty file)

- [ ] **Step 2: Write the tests**

```python
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
```

- [ ] **Step 3: Run the tests**

```bash
cd backend
uv run pytest tests/api/test_auth.py -v
```

Expected: `6 passed`.

- [ ] **Step 4: Commit**

```bash
cd ..
git add backend/tests/api/__init__.py backend/tests/api/test_auth.py
git commit -m "test: cover auth signup/login/me endpoints (priority 4)"
```

---

### Task 7: LLM provider tests (priority 5)

**Files:**
- Create: `backend/tests/services/test_llm_providers.py`

**Interfaces:**
- Consumes: `get_llm_provider(settings, *, http_client)` from
  `app.services.llm_providers.factory` (existing); `AnthropicProvider`, `GeminiProvider` from
  `app.services.llm_providers.anthropic_provider`/`.gemini_provider` (existing); `Message` TypedDict
  from `app.services.llm_providers.protocol` (existing).

- [ ] **Step 1: Write the tests**

```python
"""Priority 5 coverage: app/services/llm_providers/. All requests mocked -
no live Anthropic or Gemini call, ever. AnthropicProvider is REST-over-httpx
(mocked via httpx.MockTransport); GeminiProvider uses the google-genai SDK,
which doesn't go through httpx at all, so its client method is patched
directly instead (same technique used for live-Gemini-blocked verification
in an earlier session - see backend/README.md's "Provider-Agnostic LLM
Layer" section).
"""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.core.config import Settings
from app.services.llm_providers.anthropic_provider import AnthropicProvider
from app.services.llm_providers.factory import get_llm_provider
from app.services.llm_providers.gemini_provider import GeminiProvider


def _anthropic_mock_transport(response_text: str = "mocked response") -> httpx.MockTransport:
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": response_text}]},
        )

    transport = httpx.MockTransport(handler)
    transport.captured_requests = captured_requests  # type: ignore[attr-defined]
    return transport


@pytest.mark.asyncio(loop_scope="session")
async def test_anthropic_provider_builds_correct_request_and_parses_response():
    settings = Settings(anthropic_api_key="test-key", llm_provider="anthropic")
    transport = _anthropic_mock_transport("hello from anthropic")

    async with httpx.AsyncClient(transport=transport) as client:
        provider = AnthropicProvider(settings, http_client=client)
        result = await provider.complete(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hi."},
            ],
            "fast",
        )

    assert result == "hello from anthropic"
    request = transport.captured_requests[0]  # type: ignore[attr-defined]
    assert request.url.path == "/v1/messages"
    assert request.headers["x-api-key"] == "test-key"
    body = json.loads(request.content)
    assert body["model"] == settings.anthropic_fast_model
    assert body["system"] == "You are a helpful assistant."
    assert body["messages"] == [{"role": "user", "content": "Say hi."}]


@pytest.mark.asyncio(loop_scope="session")
async def test_anthropic_provider_uses_strong_model_for_strong_tier():
    settings = Settings(anthropic_api_key="test-key")
    transport = _anthropic_mock_transport()

    async with httpx.AsyncClient(transport=transport) as client:
        provider = AnthropicProvider(settings, http_client=client)
        await provider.complete([{"role": "user", "content": "hi"}], "strong")

    request = transport.captured_requests[0]  # type: ignore[attr-defined]
    body = json.loads(request.content)
    assert body["model"] == settings.anthropic_strong_model


@pytest.mark.asyncio(loop_scope="session")
async def test_gemini_provider_parses_response_text():
    settings = Settings(gemini_api_key="test-key", llm_provider="gemini")
    provider = GeminiProvider(settings)

    fake_response = type("FakeResponse", (), {"text": "hello from gemini"})()
    with patch.object(
        provider._client.aio.models, "generate_content", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = fake_response
        result = await provider.complete([{"role": "user", "content": "hi"}], "fast")

    assert result == "hello from gemini"
    _, call_kwargs = mock_generate.call_args
    assert call_kwargs["model"] == settings.gemini_fast_model


@pytest.mark.asyncio(loop_scope="session")
async def test_get_llm_provider_selects_by_config():
    settings_gemini = Settings(llm_provider="gemini", gemini_api_key="test-key")
    settings_anthropic = Settings(llm_provider="anthropic", anthropic_api_key="test-key")

    async with httpx.AsyncClient() as client:
        assert isinstance(get_llm_provider(settings_gemini, http_client=client), GeminiProvider)
        assert isinstance(
            get_llm_provider(settings_anthropic, http_client=client), AnthropicProvider
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_get_llm_provider_rejects_unknown_provider():
    settings = Settings(llm_provider="not-a-real-provider")
    async with httpx.AsyncClient() as client:
        with pytest.raises(RuntimeError):
            get_llm_provider(settings, http_client=client)
```

- [ ] **Step 2: Run the tests**

```bash
cd backend
uv run pytest tests/services/test_llm_providers.py -v
```

Expected: `5 passed`. (No real `GEMINI_API_KEY`/`ANTHROPIC_API_KEY` needed - both providers accept
any string as their key when constructed directly with `Settings(...)`, and no live call is ever
made.)

- [ ] **Step 3: Commit**

```bash
cd ..
git add backend/tests/services/test_llm_providers.py
git commit -m "test: cover LLM provider selection + request/response handling (priority 5)"
```

---

### Task 8: Graph node tool-failure-as-data test (priority 6)

**Files:**
- Create: `backend/tests/agent/__init__.py`
- Create: `backend/tests/agent/test_graph_tool_failure.py`

**Interfaces:**
- Consumes: `retrieve_context_node` from `app.agent.graph` (existing); `BaseTool`, `ToolContext` from
  `app.agent.tools.base` (existing); `ToolRegistry` from `app.agent.tools.registry` (existing).

- [ ] **Step 1: Create `backend/tests/agent/__init__.py`** (empty file)

- [ ] **Step 2: Write the test**

```python
"""Priority 6 coverage: a failing tool degrades a node to
tool_logs status=failed + graph status=partial, without raising -
app/agent/graph.py's core "tool failures are data, not exceptions" pattern
(see CLAUDE.md's "Conventions to follow when editing").
"""

import pytest
from pydantic import BaseModel

from app.agent.graph import retrieve_context_node
from app.agent.tools.base import BaseTool, ToolContext
from app.agent.tools.registry import ToolRegistry
from app.core.config import get_settings


class _AlwaysFailsTool(BaseTool):
    name = "destination_context_retriever"
    description = "Test double that always raises."
    input_model = BaseModel

    async def arun(self, payload: BaseModel, context: ToolContext) -> BaseModel:
        raise RuntimeError("simulated RAG retrieval failure")


@pytest.mark.asyncio(loop_scope="session")
async def test_tool_failure_produces_failed_tool_log_and_partial_status():
    registry = ToolRegistry()
    registry.register(_AlwaysFailsTool())
    context = ToolContext(settings=get_settings(), resources={}, session=None, http_client=None)

    state = {
        "prompt": "a trip to nowhere",
        "retrieval_top_k": 5,
        "status": "completed",
        "response_sections": [],
        "tool_logs": [],
        "tool_registry": registry,
        "tool_context": context,
    }

    result = await retrieve_context_node(state)

    assert result["status"] == "partial"
    failed_logs = [log for log in result["tool_logs"] if log["status"] == "failed"]
    assert len(failed_logs) == 1
    assert failed_logs[0]["tool_name"] == "destination_context_retriever"
    assert "simulated RAG retrieval failure" in failed_logs[0]["output_payload"]


@pytest.mark.asyncio(loop_scope="session")
async def test_missing_tool_runtime_also_degrades_gracefully():
    """tool_registry=None (shared services unavailable) is the other real
    "can't run this tool" path this node handles - also data, not an
    exception.
    """
    state = {
        "prompt": "a trip to nowhere",
        "retrieval_top_k": 5,
        "status": "completed",
        "response_sections": [],
        "tool_logs": [],
        "tool_registry": None,
        "tool_context": None,
    }

    result = await retrieve_context_node(state)

    assert result["status"] == "partial"
    assert result["tool_logs"][0]["status"] == "failed"
```

- [ ] **Step 3: Run the tests**

```bash
cd backend
uv run pytest tests/agent/test_graph_tool_failure.py -v
```

Expected: `2 passed`.

- [ ] **Step 4: Commit**

```bash
cd ..
git add backend/tests/agent/__init__.py backend/tests/agent/test_graph_tool_failure.py
git commit -m "test: cover tool-failure-as-data pattern in the LangGraph pipeline (priority 6)"
```

---

### Task 9: Lint/type gates — ruff + mypy config

**Files:**
- Modify: `backend/pyproject.toml` (`[tool.ruff]`, `[tool.mypy]`, `[[tool.mypy.overrides]]`)
- Modify: `backend/app/services/destination_ingestion.py` (drop the one real unused import)

**Interfaces:**
- Produces: `uv run ruff check .` and `uv run mypy app` both exit 0 from `backend/` - Task 10's CI
  workflow runs exactly these two commands.

- [ ] **Step 1: Fix the one real ruff finding**

In `backend/app/services/destination_ingestion.py`, line 23, remove the unused import:

```python
from app.services.voyage_embeddings import build_text_batches, embed_texts, estimate_text_tokens
```

becomes:

```python
from app.services.voyage_embeddings import build_text_batches, embed_texts
```

- [ ] **Step 2: Add ruff config**

Add to `backend/pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py314"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
```

- [ ] **Step 3: Verify ruff is clean**

```bash
cd backend
uv run ruff check .
```

Expected: `All checks passed!`

- [ ] **Step 4: Add mypy config with the documented pre-existing-error override**

Add to `backend/pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.14"
explicit_package_bases = true
ignore_missing_imports = true
warn_unused_ignores = true

# app/agent/graph.py's LangGraph node functions are annotated `-> TripPlannerState`
# but, by LangGraph convention, actually return a partial dict (only the keys
# that changed) - a real, pre-existing, intentional pattern (see
# app/agent/graph.py's node functions), not a bug this test-suite PR should
# refactor. Scoped narrowly to this one module so mypy still gates every
# other module at normal strictness.
[[tool.mypy.overrides]]
module = "app.agent.graph"
disable_error_code = ["typeddict-item", "arg-type", "attr-defined", "return-value"]
```

- [ ] **Step 5: Verify mypy is clean**

```bash
cd backend
uv run mypy app
```

Expected: `Success: no issues found in <N> source files` (the pre-existing `app/services/llm.py`
`tourism_level` Literal mismatch and all `app/agent/graph.py` errors are resolved/suppressed by the
steps above - if `llm.py` still reports an error, read it: it's
`Argument "tourism_level" to "TravelStylePredictionRequest" has incompatible type "str"; expected
"Literal['low', 'medium', 'high']"` at line 435 - if present, fix by changing the literal string
passed there to one of the three allowed values, since that's a real, trivially-fixable mismatch
rather than a structural pattern worth suppressing).

- [ ] **Step 6: Commit**

```bash
cd ..
git add backend/pyproject.toml backend/app/services/destination_ingestion.py
git commit -m "chore(lint): configure ruff + mypy, fix the one real unused-import finding"
```

---

### Task 10: CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:** none (this is the terminal consumer of every fixture/config from Tasks 0-9).

- [ ] **Step 1: Create `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:0.8.2-pg17
        env:
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: smart_travel_assistant_test
        ports:
          - 5432:5432
        options: >-
          --health-cmd "pg_isready -U postgres"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    env:
      DATABASE_URL: postgresql+asyncpg://postgres:postgres@localhost:5432/smart_travel_assistant_test

    defaults:
      run:
        working-directory: backend

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true

      - name: Install dependencies
        run: uv sync --all-extras --dev

      - name: Apply migrations
        run: uv run alembic upgrade head

      - name: Run tests with coverage
        run: uv run pytest --cov --cov-report=term-missing --cov-report=xml

      - name: Upload coverage summary
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: coverage-report
          path: backend/coverage.xml

  lint:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: backend
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true

      - name: Install dependencies
        run: uv sync --all-extras --dev

      - name: Ruff
        run: uv run ruff check .

      - name: mypy
        run: uv run mypy app
```

Note: the CI workflow's `_ensure_test_database_exists()` step inside `conftest.py`'s
`_test_database_ready` fixture is a no-op here - the `postgres` service already creates
`smart_travel_assistant_test` via `POSTGRES_DB`, so the `CREATE DATABASE` branch in
`_ensure_test_database_exists()` finds it already exists and skips straight to `alembic upgrade head`
(also run explicitly as its own CI step, which is fine - `alembic upgrade head` is idempotent).

- [ ] **Step 2: Verify the workflow YAML is syntactically valid**

```bash
cd "C:/Users/Kayan/OneDrive/Desktop/projects/sef_projects/smart_travel_assistant"
uv run --with pyyaml python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('valid YAML')"
```

Expected: `valid YAML`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions workflow (postgres+pgvector service, pytest+coverage, ruff, mypy)"
```

---

### Task 11: Documentation — README + CLAUDE.md

**Files:**
- Modify: `backend/README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add a "Running Tests" section to `backend/README.md`**

Add near the top of `backend/README.md` (after the ML Workflow section header, or any other
reasonable top-level location):

````markdown
## Running Tests

![CI](https://github.com/kayan2004/smart-travel-planner/actions/workflows/ci.yml/badge.svg)

```powershell
# One-time: create the test database (same Postgres the dev stack uses)
docker exec smart_travel_assistant-db-1 psql -U postgres -c "CREATE DATABASE smart_travel_assistant_test"

# From backend/
$env:DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/smart_travel_assistant_test"
uv run pytest --cov --cov-report=term-missing
```

Tests never touch the dev database (`smart_travel_assistant`) - `conftest.py` asserts `"test"` is in
the configured `DATABASE_URL` before running anything, and creates + migrates
`smart_travel_assistant_test` automatically on first run if it doesn't exist yet. Isolation between
tests is **truncate-after**, not transactional rollback - several services under test commit
internally (`persist_recommendation_slate`, `submit_feedback`), which would silently defeat a
rollback-based recipe.

All external HTTP (Voyage, Anthropic, Open-Meteo, Discord) is mocked via `httpx.MockTransport`; the
`google-genai` SDK (Gemini) is mocked by patching its client method directly, since that SDK doesn't
go through httpx. No live API calls happen in the test suite, ever.

Coverage target is ~70% of `app/services` + `app/agent` - a target, not a 100%-or-fail gate.

CI (`.github/workflows/ci.yml`) runs the same suite against a `postgres`/`pgvector` service
container on every push/PR, plus `ruff check` and `mypy app` as separate jobs.
````

- [ ] **Step 2: Update `CLAUDE.md`'s "Known gaps" section**

In `CLAUDE.md`, find:

```markdown
## Known gaps (see README "Known Gaps" for the full list)

No automated tests or CI, no LangSmith/tracing, no per-step token/cost logging, no webhook
retry-with-backoff. Be aware of these when asked to "add tests" or "wire up retries" — there's no
existing pattern to extend, you'd be establishing the first one.
```

Replace with:

```markdown
## Known gaps (see README "Known Gaps" for the full list)

No LangSmith/tracing, no per-step token/cost logging, no webhook retry-with-backoff. Be aware of
these when asked to "wire up retries" — there's no existing pattern for those to extend.

**Automated tests + CI now exist** (`backend/tests/`, `.github/workflows/ci.yml`) - pytest +
pytest-asyncio against a dedicated test Postgres (never the dev DB), truncate-based isolation
(rollback-based isolation does NOT work here - several services commit internally), every external
HTTP boundary mocked. `backend/tests/conftest.py` is the pattern to extend for new test coverage;
see `backend/README.md`'s "Running Tests" section for the full write-up.
```

- [ ] **Step 3: Commit**

```bash
git add backend/README.md CLAUDE.md
git commit -m "docs: document the test suite, CI, and update the no-tests known-gap"
```

---

### Task 12: Final review and finish the branch

- [ ] **Step 1: Run the full suite once, end to end, from a clean test database**

```bash
cd "C:/Users/Kayan/OneDrive/Desktop/projects/sef_projects/smart_travel_assistant/backend"
docker exec smart_travel_assistant-db-1 psql -U postgres -c "DROP DATABASE IF EXISTS smart_travel_assistant_test"
$env:DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/smart_travel_assistant_test"
uv run pytest --cov --cov-report=term-missing -v
```

Expected: every test from Tasks 3-8 passes (`26` total: 7+3+3+6+5+2), coverage report prints with
`app/services/destination_recommendations.py`, `app/services/feedback.py`,
`app/services/recommendation_persistence.py`, `app/services/llm_providers/*` all showing non-trivial
coverage.

- [ ] **Step 2: Run ruff + mypy once more on the full branch**

```bash
uv run ruff check .
uv run mypy app
```

Expected: both clean (`All checks passed!` / `Success: no issues found`).

- [ ] **Step 3: Review the full diff against `main`**

```bash
cd ..
git diff main...feat/pytest-suite-and-ci --stat
```

Confirm the file list matches this plan's "File Structure" section and no unrelated files snuck in.

- [ ] **Step 4: Use `superpowers:finishing-a-development-branch`** to present the merge/PR/keep/discard
  menu and complete the branch per whichever option is chosen (this user's standing default is
  "merge to main locally, then push").

---
