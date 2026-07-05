# Recommendation Slate Persistence + Feedback Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task, inline in this session on a feature branch (no subagent dispatch). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist the full ranked recommendation slate (not just the top result) that the pre-filter+cosine node already returns, and add an anonymous thumbs up/down feedback loop keyed by a client-generated session UUID, so each `recommendations` row plus its `feedback.verdict` becomes a labeled training example for a future learning-to-rank model.

**Architecture:** `recommendations`/`feedback` are existing schema-only tables (from the 2026-07-04 session). The agent_run row must exist (its `id` is the FK target) before the slate can be persisted, so persistence happens in `app/services/agent_runs.py::create_agent_run` right after the `agent_run` row is committed — mirroring how `tool_logs` are already persisted there, and logged as its own `tool_logs` entry so a persistence failure degrades gracefully instead of failing the user-facing response. A new `POST /feedback` endpoint (no auth — anonymous, `session_uuid`-keyed) does an idempotent Postgres `INSERT ... ON CONFLICT (recommendation_id, session_uuid) DO UPDATE` upsert. The frontend generates and persists a `session_uuid` in `localStorage`, renders the ranked slate with thumbs up/down buttons, and POSTs on click.

**Tech Stack:** FastAPI, SQLAlchemy 2.x async (asyncpg), Alembic, Postgres `ON CONFLICT` upsert, React 19 + TypeScript, `crypto.randomUUID()`.

## Global Constraints

- Async everywhere: every new DB call is `await`ed on an `AsyncSession`, no blocking calls.
- `features` JSONB is captured once, at recommend time, exactly as the recommendation node already computed it — never recomputed or overwritten later.
- Persist the **full slate** returned by the recommendation node (currently `limit=3`, so 3 rows per run), not just the top result.
- Feedback dedup: **one verdict per `(session_uuid, recommendation_id)`**, enforced by a real Postgres unique constraint (not just application-level checking) — re-submitting updates the existing row's `verdict` rather than inserting a duplicate.
- `feedback.channel` defaults to `"web"` (a plain string column, not an enum) so future non-web channels (e.g. a CLI or Discord reaction) can reuse the same table without a migration.
- No PII beyond `session_uuid` (a random client-generated UUID, never tied to a `users` row) — the feedback endpoint takes no auth token and must not require one.
- Slate persistence failures are data, not fatal errors: wrap the persist call in `try/except`, log a `tool_logs` entry with `status="failed"` on exception (after `session.rollback()`), and let the user-facing response continue — mirroring the existing Discord-webhook-delivery pattern in the same file.
- This project has no automated test suite and none should be introduced (see `CLAUDE.md`'s "Known gaps"). Verification is manual: `uv run python -` scratch scripts against the real local dev Postgres (already running, already has 219 real destinations and 7 real users), each cleaning up the scratch rows it creates so the dev DB isn't left with fake data.
- Execute this plan **inline, in this session, on a new feature branch** — no subagent dispatch (see the plan's commit history for why: a prior subagent-driven run had an implementer commit into the wrong checkout).

---

## File Structure

**Create:**
- `backend/alembic/versions/<new>_add_feedback_channel_and_unique_constraint.py` — adds `feedback.channel` and the `(recommendation_id, session_uuid)` unique constraint.
- `backend/app/schemas/recommendation_read.py` — `RecommendationRead`, the persisted-row read shape (with joined destination name/country for display).
- `backend/app/schemas/feedback.py` — `FeedbackCreate`/`FeedbackRead`.
- `backend/app/services/recommendation_persistence.py` — `persist_recommendation_slate()` (write) and `get_recommendations_for_agent_run()` (read, joins to `destinations` for display fields). Deliberately **not** named `recommendations.py` — that filename was retired earlier this session specifically because reusing a name for an unrelated purpose caused confusion; don't resurrect it.
- `backend/app/services/feedback.py` — `submit_feedback()`, the idempotent upsert.
- `backend/app/api/routes/feedback.py` — `POST /feedback`, no auth dependency.

**Modify:**
- `backend/app/db/models/feedback.py` — add the `channel` column.
- `backend/app/agent/planner.py` — `PlannerResult` gains `recommended_destinations: list[dict[str, Any]]`.
- `backend/app/services/agent_runs.py` — persist the slate (non-fatal) right after the `agent_run` row is committed.
- `backend/app/schemas/agent_runs.py` — `AgentRunRead` gains `recommendations: list[RecommendationRead]`.
- `backend/app/api/routes/agent_runs.py` — assemble the response manually (not via a single `model_validate(agent_run)` call — see Task 4's note on why) so it includes the joined recommendation rows.
- `backend/main.py` — register the new feedback router.
- `frontend/src/types.ts` — `RecommendationRead`, `RecommendationFeatures`, `FeedbackVerdict`, `FeedbackRead`; extend `AgentRunRead`.
- `frontend/src/lib/api.ts` — `submitFeedback()`.
- `frontend/src/App.tsx` — session UUID generation/persistence, a new "Recommendations" panel with thumbs up/down.
- `frontend/src/App.css` — minimal styling for the feedback buttons, reusing existing class conventions.
- `backend/README.md` — new "Feedback Data Model" section (schema, endpoint contract, slate→training-row mapping).
- `README.md` (repo root) — brief mention in "Persistence" that recommendations/feedback are now populated.

---

### Task 0: Create the feature branch

**Files:** none (git only)

- [ ] **Step 1: Confirm you're on a clean `main` and create the branch**

```bash
cd "C:/Users/Kayan/OneDrive/Desktop/projects/sef_projects/smart_travel_assistant"
git status --short
git checkout -b feat/recommendation-slate-persistence-and-feedback
```

Expected: clean status (no output from `git status --short`), then a confirmation you're on the new branch.

---

### Task 1: Feedback migration — `channel` column + unique constraint

**Files:**
- Create: `backend/alembic/versions/b1f4a9d3e7c2_add_feedback_channel_and_unique_constraint.py`
- Modify: `backend/app/db/models/feedback.py`

**Interfaces:**
- Produces: `feedback.channel` (String(50), NOT NULL, server default `'web'`), and a named unique constraint `uq_feedback_recommendation_session` on `(recommendation_id, session_uuid)`. Task 5's upsert targets this constraint **by name**.

- [ ] **Step 1: Confirm current alembic head and that `feedback` is empty**

Run (from `backend/`):

```bash
uv run alembic current
docker exec smart_travel_assistant-db-1 psql -U postgres -d smart_travel_assistant -c "SELECT count(*) FROM feedback;"
```

Expected: `a7c3e5f19d02 (head)` and `count = 0`. If `feedback` is not empty, stop and report back rather than adding a unique constraint that might fail on duplicates — do not delete rows to make it pass.

- [ ] **Step 2: Write the migration**

Create `backend/alembic/versions/b1f4a9d3e7c2_add_feedback_channel_and_unique_constraint.py`:

```python
"""add channel column and unique constraint to feedback

Revision ID: b1f4a9d3e7c2
Revises: a7c3e5f19d02
Create Date: 2026-07-06 09:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'b1f4a9d3e7c2'
down_revision: Union[str, Sequence[str], None] = 'a7c3e5f19d02'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "feedback",
        sa.Column(
            "channel",
            sa.String(length=50),
            nullable=False,
            server_default="web",
        ),
    )
    op.create_unique_constraint(
        "uq_feedback_recommendation_session",
        "feedback",
        ["recommendation_id", "session_uuid"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(
        "uq_feedback_recommendation_session", "feedback", type_="unique"
    )
    op.drop_column("feedback", "channel")
```

- [ ] **Step 2: Update the `Feedback` ORM model**

In `backend/app/db/models/feedback.py`, add `String` to the existing `sqlalchemy` import and add the `channel` column between `verdict` and `created_at`:

```python
from sqlalchemy import DateTime, ForeignKey, SmallInteger, String, func
```

```python
    verdict: Mapped[int] = mapped_column(SmallInteger, nullable=False)  # +1 / -1
    # Plain string, not an enum - lets future non-web channels (a CLI, a
    # Discord reaction) reuse this table without a migration.
    channel: Mapped[str] = mapped_column(String(50), nullable=False, server_default="web")
    created_at: Mapped[datetime] = mapped_column(
```

- [ ] **Step 3: Apply the migration and inspect the result**

Run (from `backend/`):

```bash
uv run alembic upgrade head
uv run alembic current
docker exec smart_travel_assistant-db-1 psql -U postgres -d smart_travel_assistant -c "\d feedback"
```

Expected: `b1f4a9d3e7c2 (head)`, and the `\d feedback` output shows a `channel` column (`character varying(50) not null default 'web'`) and a `"uq_feedback_recommendation_session" UNIQUE CONSTRAINT, btree (recommendation_id, session_uuid)` line.

- [ ] **Step 4: Verify reversibility**

Run (from `backend/`):

```bash
uv run alembic downgrade -1
uv run alembic current
uv run alembic upgrade head
uv run alembic current
```

Expected: first `current` shows `a7c3e5f19d02 (head)` (downgrade succeeded), second shows `b1f4a9d3e7c2 (head)` again (re-upgrade succeeded) with no errors either direction.

- [ ] **Step 5: Verify the ORM model matches**

Run (from `backend/`):

```bash
uv run python -c "
from app.db.models.feedback import Feedback
columns = Feedback.__table__.columns.keys()
assert 'channel' in columns, columns
print('Feedback model OK:', columns)
"
```

Expected: prints the column list including `channel`, no traceback.

- [ ] **Step 6: Commit**

```bash
git add backend/alembic/versions/b1f4a9d3e7c2_add_feedback_channel_and_unique_constraint.py backend/app/db/models/feedback.py
git commit -m "feat(feedback): add channel column and dedup unique constraint"
```

---

### Task 2: Read/write schemas

**Files:**
- Create: `backend/app/schemas/recommendation_read.py`
- Create: `backend/app/schemas/feedback.py`

**Interfaces:**
- Produces: `RecommendationRead(id: int, destination_id: uuid.UUID, destination_name: str, country: str, rank_position: int, score: float, features: dict, created_at: datetime)`. `FeedbackCreate(recommendation_id: int, session_uuid: uuid.UUID, verdict: Literal[1, -1])`. `FeedbackRead(id: int, recommendation_id: int, session_uuid: uuid.UUID, verdict: int, channel: str, created_at: datetime)`. Task 3 constructs `RecommendationRead` instances directly (not via `model_validate` on an ORM object — see Task 3's note). Task 4 imports `RecommendationRead` into `AgentRunRead`. Task 5 imports both `FeedbackCreate`/`FeedbackRead`.

- [ ] **Step 1: Write the recommendation read schema**

Create `backend/app/schemas/recommendation_read.py`:

```python
import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class RecommendationRead(BaseModel):
    id: int
    destination_id: uuid.UUID
    destination_name: str
    country: str
    rank_position: int
    score: float
    features: dict
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
```

- [ ] **Step 2: Write the feedback schemas**

Create `backend/app/schemas/feedback.py`:

```python
import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class FeedbackCreate(BaseModel):
    recommendation_id: int
    session_uuid: uuid.UUID
    verdict: Literal[1, -1]


class FeedbackRead(BaseModel):
    id: int
    recommendation_id: int
    session_uuid: uuid.UUID
    verdict: int
    channel: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
```

- [ ] **Step 3: Verify both import and validate**

Run (from `backend/`):

```bash
uv run python - <<'PY'
import uuid
from datetime import datetime, timezone

from app.schemas.feedback import FeedbackCreate, FeedbackRead
from app.schemas.recommendation_read import RecommendationRead

feedback_in = FeedbackCreate(
    recommendation_id=1, session_uuid=uuid.uuid4(), verdict=1
)
assert feedback_in.verdict == 1

feedback_out = FeedbackRead(
    id=1,
    recommendation_id=1,
    session_uuid=feedback_in.session_uuid,
    verdict=1,
    channel="web",
    created_at=datetime.now(timezone.utc),
)
assert feedback_out.channel == "web"

recommendation = RecommendationRead(
    id=1,
    destination_id=uuid.uuid4(),
    destination_name="Bali",
    country="Indonesia",
    rank_position=1,
    score=0.81,
    features={"cosine_sim": 0.81, "tag_match_count": 0, "budget_delta": None, "region_match": True},
    created_at=datetime.now(timezone.utc),
)
assert recommendation.destination_name == "Bali"

try:
    FeedbackCreate(recommendation_id=1, session_uuid=uuid.uuid4(), verdict=2)
    raise SystemExit("expected a validation error for verdict=2")
except Exception as exc:
    assert "verdict" in str(exc) or "literal" in str(exc).lower()

print("schemas OK")
PY
```

Expected: prints `schemas OK`, no traceback.

- [ ] **Step 4: Commit**

```bash
git add backend/app/schemas/recommendation_read.py backend/app/schemas/feedback.py
git commit -m "feat(recommendations): add RecommendationRead and feedback schemas"
```

---

### Task 3: Recommendation slate persistence service

**Files:**
- Create: `backend/app/services/recommendation_persistence.py`

**Interfaces:**
- Consumes: `Recommendation` (`app.db.models.recommendation`, fields `id`, `agent_run_id`, `destination_id: uuid.UUID`, `rank_position: int`, `score: float`, `features: dict`, `created_at`, `deleted_at`); `Destination` (`app.db.models.destination`, fields `id: uuid.UUID`, `name: str`, `country: str`); `RecommendationRead` from Task 2.
- Produces: `async def persist_recommendation_slate(session: AsyncSession, agent_run_id: int, recommended_destinations: list[dict]) -> list[Recommendation]` and `async def get_recommendations_for_agent_run(session: AsyncSession, agent_run_id: int) -> list[RecommendationRead]`. Task 4 calls both.

- [ ] **Step 1: Write the service**

Create `backend/app/services/recommendation_persistence.py`:

```python
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.destination import Destination
from app.db.models.recommendation import Recommendation
from app.schemas.recommendation_read import RecommendationRead


async def persist_recommendation_slate(
    session: AsyncSession,
    agent_run_id: int,
    recommended_destinations: list[dict],
) -> list[Recommendation]:
    if not recommended_destinations:
        return []

    rows = [
        Recommendation(
            agent_run_id=agent_run_id,
            destination_id=uuid.UUID(item["destination_id"]),
            rank_position=item["rank_position"],
            score=item["score"],
            features=item["features"],
        )
        for item in recommended_destinations
    ]
    session.add_all(rows)
    await session.commit()
    return rows


async def get_recommendations_for_agent_run(
    session: AsyncSession,
    agent_run_id: int,
) -> list[RecommendationRead]:
    statement = (
        select(Recommendation, Destination.name, Destination.country)
        .join(Destination, Recommendation.destination_id == Destination.id)
        .where(Recommendation.agent_run_id == agent_run_id)
        .where(Recommendation.deleted_at.is_(None))
        .order_by(Recommendation.rank_position)
    )
    rows = (await session.execute(statement)).all()
    return [
        RecommendationRead(
            id=recommendation.id,
            destination_id=recommendation.destination_id,
            destination_name=name,
            country=country,
            rank_position=recommendation.rank_position,
            score=recommendation.score,
            features=recommendation.features,
            created_at=recommendation.created_at,
        )
        for recommendation, name, country in rows
    ]
```

Note for context (not a step to act on): `Recommendation` and `Destination` live on two separate SQLAlchemy `DeclarativeBase` registries (see `app/db/models/destination.py`'s docstring) — this is why the query uses an explicit `.join(Destination, Recommendation.destination_id == Destination.id)` with a manual `onclause` instead of a `relationship()`-based join. That's deliberate and correct; don't try to add a `relationship()` between them.

- [ ] **Step 2: Verify against the real dev database**

This creates one throwaway `agent_runs` row (tied to a real existing user) and two throwaway `destinations`-backed recommendation rows, round-trips them through both functions, then deletes everything it created. Run (from `backend/`):

```bash
uv run python - <<'PY'
import asyncio
import uuid

from sqlalchemy import delete, select

from app.core.config import get_settings
from app.db.models.agent_run import AgentRun
from app.db.models.destination import Destination
from app.db.models.recommendation import Recommendation
from app.db.models.user import User
from app.db.session import create_db_engine, create_session_factory
from app.services.recommendation_persistence import (
    get_recommendations_for_agent_run,
    persist_recommendation_slate,
)


async def main() -> None:
    settings = get_settings()
    engine = create_db_engine(settings)
    session_factory = create_session_factory(engine)

    async with session_factory() as session:
        user = (await session.execute(select(User).limit(1))).scalar_one()
        destinations = (
            await session.execute(select(Destination).limit(2))
        ).scalars().all()
        assert len(destinations) == 2, "need at least 2 destinations in the dev DB"

        agent_run = AgentRun(
            user_id=user.id,
            prompt="scratch verification run",
            response="scratch",
            status="completed",
        )
        session.add(agent_run)
        await session.commit()
        await session.refresh(agent_run)

        recommended_destinations = [
            {
                "destination_id": str(destinations[0].id),
                "rank_position": 1,
                "score": 0.91,
                "features": {
                    "cosine_sim": 0.91,
                    "tag_match_count": 0,
                    "budget_delta": None,
                    "region_match": True,
                },
            },
            {
                "destination_id": str(destinations[1].id),
                "rank_position": 2,
                "score": 0.77,
                "features": {
                    "cosine_sim": 0.77,
                    "tag_match_count": 0,
                    "budget_delta": None,
                    "region_match": True,
                },
            },
        ]

        persisted = await persist_recommendation_slate(
            session, agent_run.id, recommended_destinations
        )
        assert len(persisted) == 2, persisted
        assert all(row.id is not None for row in persisted), persisted

        read_back = await get_recommendations_for_agent_run(session, agent_run.id)
        assert len(read_back) == 2, read_back
        assert read_back[0].rank_position == 1
        assert read_back[0].destination_name == destinations[0].name
        assert read_back[1].rank_position == 2
        assert read_back[0].features["cosine_sim"] == 0.91

        # Clean up the scratch rows - never leave fake data in the dev DB.
        await session.execute(
            delete(Recommendation).where(Recommendation.agent_run_id == agent_run.id)
        )
        await session.execute(delete(AgentRun).where(AgentRun.id == agent_run.id))
        await session.commit()

    await engine.dispose()
    print("persistence service OK")


asyncio.run(main())
PY
```

Expected: prints `persistence service OK`, no traceback. Confirm cleanup worked:

```bash
docker exec smart_travel_assistant-db-1 psql -U postgres -d smart_travel_assistant -c "SELECT prompt FROM agent_runs WHERE prompt = 'scratch verification run';"
```

Expected: `(0 rows)`.

- [ ] **Step 3: Commit**

```bash
git add backend/app/services/recommendation_persistence.py
git commit -m "feat(recommendations): add slate persistence and agent-run read service"
```

---

### Task 4: Wire persistence into the agent-run flow

**Files:**
- Modify: `backend/app/agent/planner.py`
- Modify: `backend/app/services/agent_runs.py`
- Modify: `backend/app/schemas/agent_runs.py`
- Modify: `backend/app/api/routes/agent_runs.py`

**Interfaces:**
- Consumes: `persist_recommendation_slate`/`get_recommendations_for_agent_run` (Task 3), `RecommendationRead` (Task 2), `create_tool_log` (`app.services.tool_logs`, already exists: `async def create_tool_log(session, agent_run, *, tool_name, input_payload, output_payload, status="completed") -> ToolLog`).
- Produces: `PlannerResult.recommended_destinations: list[dict[str, Any]]`; `AgentRunRead.recommendations: list[RecommendationRead]`. Task 6 (frontend) mirrors this exact response shape in `types.ts`.

- [ ] **Step 1: Add `recommended_destinations` to `PlannerResult`**

In `backend/app/agent/planner.py`, add `from typing import Any` to the imports (alongside the existing `from dataclasses import dataclass`), then update the dataclass and its construction:

```python
from dataclasses import dataclass
from typing import Any

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
    recommended_destinations: list[dict[str, Any]]
```

Then update the `return PlannerResult(...)` at the end of `run_trip_planner` to add one more field:

```python
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
        recommended_destinations=list(final_state.get("recommended_destinations") or []),
    )
```

- [ ] **Step 2: Persist the slate in `create_agent_run`, non-fatally**

In `backend/app/services/agent_runs.py`, add these imports:

```python
from app.services.recommendation_persistence import persist_recommendation_slate
```

Then, immediately after the existing `tool_logs` loop (`for tool_log in planner_result.tool_logs: await create_tool_log(...)`) and before the Discord webhook block, insert:

```python
    try:
        await persist_recommendation_slate(
            session, agent_run.id, planner_result.recommended_destinations
        )
        await create_tool_log(
            session,
            agent_run,
            tool_name="recommendation_persistence",
            input_payload=f"{len(planner_result.recommended_destinations)} slate row(s)",
            output_payload="Recommendation slate persisted successfully.",
            status="completed",
        )
    except Exception as exc:
        await session.rollback()
        await create_tool_log(
            session,
            agent_run,
            tool_name="recommendation_persistence",
            input_payload=f"{len(planner_result.recommended_destinations)} slate row(s)",
            output_payload=(
                f"Recommendation slate persistence failed: {type(exc).__name__}: {exc}"
            ),
            status="failed",
        )
```

Finally, update the existing refresh call at the end of the function from:

```python
    await session.refresh(agent_run, attribute_names=["tool_logs"])
```

to:

```python
    await session.refresh(agent_run, attribute_names=["tool_logs", "recommendations"])
```

- [ ] **Step 3: Add `recommendations` to `AgentRunRead`**

In `backend/app/schemas/agent_runs.py`, add the import and field:

```python
from app.schemas.recommendation_read import RecommendationRead
```

```python
class AgentRunRead(BaseModel):
    id: int
    user_id: int
    prompt: str
    response: str
    status: str
    created_at: datetime
    tool_logs: list[ToolLogRead] = []
    recommendations: list[RecommendationRead] = []

    model_config = ConfigDict(from_attributes=True)
```

- [ ] **Step 4: Assemble the route response manually — do not call `AgentRunRead.model_validate(agent_run)` directly**

**Why this step matters:** `agent_run.recommendations` (the ORM relationship) is a list of bare `Recommendation` objects, which do **not** have `destination_name`/`country` attributes — only `RecommendationRead` needs those (joined in from `Destination`). If you call `AgentRunRead.model_validate(agent_run)` with `recommendations` now a required-shape field, Pydantic will try to auto-populate it from `agent_run.recommendations` via `from_attributes` and raise a validation error because those attributes don't exist on the ORM object. Build the response explicitly instead, using the joined read model from Task 3.

Replace the route body in `backend/app/api/routes/agent_runs.py`:

```python
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import get_current_user
from app.db.dependencies import get_db_session
from app.db.models.user import User
from app.schemas.agent_runs import AgentRunCreate, AgentRunRead
from app.schemas.tool_logs import ToolLogRead
from app.services.agent_runs import create_agent_run
from app.services.recommendation_persistence import get_recommendations_for_agent_run
from app.agent.tools.base import ToolContext

router = APIRouter(prefix="/agent-runs", tags=["agent-runs"])


@router.post("", response_model=AgentRunRead, status_code=status.HTTP_201_CREATED)
async def create_agent_run_route(
    payload: AgentRunCreate,
    request: Request,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> AgentRunRead:
    tool_registry = request.app.state.resources.get("tool_registry")
    http_client = request.app.state.resources.get("http_client")
    tool_context = ToolContext(
        settings=request.app.state.settings,
        resources=request.app.state.resources,
        session=session,
        http_client=http_client,
    )
    agent_run = await create_agent_run(
        session,
        current_user,
        payload,
        tool_registry=tool_registry,
        tool_context=tool_context,
    )
    recommendations = await get_recommendations_for_agent_run(session, agent_run.id)
    return AgentRunRead(
        id=agent_run.id,
        user_id=agent_run.user_id,
        prompt=agent_run.prompt,
        response=agent_run.response,
        status=agent_run.status,
        created_at=agent_run.created_at,
        tool_logs=[ToolLogRead.model_validate(log) for log in agent_run.tool_logs],
        recommendations=recommendations,
    )
```

The block above is the complete replacement for the file (imports included) — the current file already imports `Request` from `fastapi` on line 1, so this is a like-for-like replacement of the whole file, not a partial edit.

- [ ] **Step 5: Verify end-to-end against the real dev database with a mocked Voyage call**

This runs the actual `create_agent_run` service (not just the graph in isolation) against the real DB, confirming: the agent_run is created, the slate is persisted, `get_recommendations_for_agent_run` returns it, and the cleanup removes everything. Run (from `backend/`):

```bash
uv run python - <<'PY'
import asyncio
import httpx
from sqlalchemy import delete, select

from app.agent.tools.base import ToolContext
from app.agent.tools.registry import build_default_tool_registry
from app.core.config import get_settings
from app.db.models.agent_run import AgentRun
from app.db.models.recommendation import Recommendation
from app.db.models.tool_log import ToolLog
from app.db.models.user import User
from app.db.session import create_db_engine, create_session_factory
from app.schemas.agent_runs import AgentRunCreate
from app.services.agent_runs import create_agent_run


def _mock_transport(request: httpx.Request) -> httpx.Response:
    if "voyageai" in str(request.url):
        return httpx.Response(200, json={"data": [{"embedding": [0.01] * 1024}]})
    return httpx.Response(503, json={"error": "no live LLM call in this smoke test"})


async def main() -> None:
    settings = get_settings()
    engine = create_db_engine(settings)
    session_factory = create_session_factory(engine)
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(_mock_transport))
    tool_registry = build_default_tool_registry()

    async with session_factory() as session:
        user = (await session.execute(select(User).limit(1))).scalar_one()
        tool_context = ToolContext(
            settings=settings, resources={}, session=session, http_client=http_client
        )
        agent_run = await create_agent_run(
            session,
            user,
            AgentRunCreate(prompt="a relaxing beach vacation on a low budget", retrieval_top_k=3),
            tool_registry=tool_registry,
            tool_context=tool_context,
        )

        assert agent_run.recommendations, "expected at least one persisted recommendation"
        persistence_logs = [
            log for log in agent_run.tool_logs if log.tool_name == "recommendation_persistence"
        ]
        assert len(persistence_logs) == 1, agent_run.tool_logs
        assert persistence_logs[0].status == "completed", persistence_logs[0].output_payload
        print(
            "persisted",
            len(agent_run.recommendations),
            "recommendation row(s) for agent_run",
            agent_run.id,
        )

        # Clean up.
        await session.execute(
            delete(Recommendation).where(Recommendation.agent_run_id == agent_run.id)
        )
        await session.execute(delete(ToolLog).where(ToolLog.agent_run_id == agent_run.id))
        await session.execute(delete(AgentRun).where(AgentRun.id == agent_run.id))
        await session.commit()

    await http_client.aclose()
    await engine.dispose()
    print("end-to-end persistence OK")


asyncio.run(main())
PY
```

Expected: prints `persisted N recommendation row(s)...` then `end-to-end persistence OK`, no traceback.

- [ ] **Step 6: Commit**

```bash
git add backend/app/agent/planner.py backend/app/services/agent_runs.py backend/app/schemas/agent_runs.py backend/app/api/routes/agent_runs.py
git commit -m "feat(agent-runs): persist recommendation slate and return it in the response"
```

---

### Task 5: Feedback endpoint

**Files:**
- Create: `backend/app/services/feedback.py`
- Create: `backend/app/api/routes/feedback.py`
- Modify: `backend/main.py`

**Interfaces:**
- Consumes: `Feedback` (Task 1), `FeedbackCreate`/`FeedbackRead` (Task 2).
- Produces: `POST /feedback` (no auth), `RecommendationNotFoundError` exception class. Task 6 (frontend) calls this exact path/body shape.

- [ ] **Step 1: Write the feedback service**

Create `backend/app/services/feedback.py`:

```python
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.feedback import Feedback
from app.schemas.feedback import FeedbackCreate, FeedbackRead


class RecommendationNotFoundError(Exception):
    """Raised when FeedbackCreate.recommendation_id does not exist."""


async def submit_feedback(
    session: AsyncSession,
    payload: FeedbackCreate,
    *,
    channel: str = "web",
) -> FeedbackRead:
    statement = (
        insert(Feedback)
        .values(
            recommendation_id=payload.recommendation_id,
            session_uuid=payload.session_uuid,
            verdict=payload.verdict,
            channel=channel,
        )
        .on_conflict_do_update(
            constraint="uq_feedback_recommendation_session",
            set_={"verdict": payload.verdict, "channel": channel},
        )
        .returning(
            Feedback.id,
            Feedback.recommendation_id,
            Feedback.session_uuid,
            Feedback.verdict,
            Feedback.channel,
            Feedback.created_at,
        )
    )

    try:
        result = await session.execute(statement)
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise RecommendationNotFoundError(
            f"Recommendation {payload.recommendation_id} does not exist."
        ) from exc

    row = result.one()
    return FeedbackRead(
        id=row.id,
        recommendation_id=row.recommendation_id,
        session_uuid=row.session_uuid,
        verdict=row.verdict,
        channel=row.channel,
        created_at=row.created_at,
    )
```

- [ ] **Step 2: Write the route**

Create `backend/app/api/routes/feedback.py`:

```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.dependencies import get_db_session
from app.schemas.feedback import FeedbackCreate, FeedbackRead
from app.services.feedback import RecommendationNotFoundError, submit_feedback

router = APIRouter(tags=["feedback"])


@router.post("/feedback", response_model=FeedbackRead, status_code=status.HTTP_200_OK)
async def submit_feedback_route(
    payload: FeedbackCreate,
    session: AsyncSession = Depends(get_db_session),
) -> FeedbackRead:
    try:
        return await submit_feedback(session, payload)
    except RecommendationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
```

No `get_current_user` dependency here, deliberately — feedback is anonymous and session-uuid-keyed, not tied to an authenticated user.

- [ ] **Step 3: Register the router**

In `backend/main.py`, add the import alongside the other route imports (keep alphabetical order with the existing block):

```python
from app.api.routes.feedback import router as feedback_router
```

And add the `include_router` call alongside the others (again, alphabetical with the existing block):

```python
    application.include_router(feedback_router)
```

- [ ] **Step 4: Verify the idempotent upsert against the real dev database**

This creates one throwaway `agent_runs` + `recommendations` row, submits feedback twice with different verdicts for the same `(recommendation_id, session_uuid)`, confirms exactly one row with the updated verdict, then tests the FK-miss path, then cleans up. Run (from `backend/`):

```bash
uv run python - <<'PY'
import asyncio
import uuid

from sqlalchemy import delete, func, select

from app.core.config import get_settings
from app.db.models.agent_run import AgentRun
from app.db.models.destination import Destination
from app.db.models.feedback import Feedback
from app.db.models.recommendation import Recommendation
from app.db.models.user import User
from app.db.session import create_db_engine, create_session_factory
from app.schemas.feedback import FeedbackCreate
from app.services.feedback import RecommendationNotFoundError, submit_feedback


async def main() -> None:
    settings = get_settings()
    engine = create_db_engine(settings)
    session_factory = create_session_factory(engine)

    async with session_factory() as session:
        user = (await session.execute(select(User).limit(1))).scalar_one()
        destination = (
            await session.execute(select(Destination).limit(1))
        ).scalar_one()

        agent_run = AgentRun(
            user_id=user.id, prompt="scratch feedback run", response="scratch", status="completed"
        )
        session.add(agent_run)
        await session.commit()
        await session.refresh(agent_run)

        recommendation = Recommendation(
            agent_run_id=agent_run.id,
            destination_id=destination.id,
            rank_position=1,
            score=0.5,
            features={"cosine_sim": 0.5, "tag_match_count": 0, "budget_delta": None, "region_match": True},
        )
        session.add(recommendation)
        await session.commit()
        await session.refresh(recommendation)

        session_uuid = uuid.uuid4()

        first = await submit_feedback(
            session,
            FeedbackCreate(recommendation_id=recommendation.id, session_uuid=session_uuid, verdict=1),
        )
        assert first.verdict == 1, first

        second = await submit_feedback(
            session,
            FeedbackCreate(recommendation_id=recommendation.id, session_uuid=session_uuid, verdict=-1),
        )
        assert second.verdict == -1, second
        assert second.id == first.id, "resubmit should update the same row, not insert a new one"

        row_count = (
            await session.execute(
                select(func.count()).select_from(Feedback).where(
                    Feedback.recommendation_id == recommendation.id
                )
            )
        ).scalar_one()
        assert row_count == 1, f"expected exactly 1 feedback row, got {row_count}"

        try:
            await submit_feedback(
                session,
                FeedbackCreate(recommendation_id=999999999, session_uuid=uuid.uuid4(), verdict=1),
            )
            raise SystemExit("expected RecommendationNotFoundError for a nonexistent recommendation_id")
        except RecommendationNotFoundError:
            pass

        # Clean up.
        await session.execute(delete(Feedback).where(Feedback.recommendation_id == recommendation.id))
        await session.execute(delete(Recommendation).where(Recommendation.id == recommendation.id))
        await session.execute(delete(AgentRun).where(AgentRun.id == agent_run.id))
        await session.commit()

    await engine.dispose()
    print("feedback upsert + dedup + FK-miss OK")


asyncio.run(main())
PY
```

Expected: prints `feedback upsert + dedup + FK-miss OK`, no traceback.

- [ ] **Step 5: Verify the app still imports and the route is registered**

Run (from `backend/`):

```bash
uv run python -c "
from main import app
paths = {route.path for route in app.routes}
assert '/feedback' in paths
print('feedback route OK')
"
```

Expected: prints `feedback route OK`.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/feedback.py backend/app/api/routes/feedback.py backend/main.py
git commit -m "feat(feedback): add idempotent POST /feedback endpoint"
```

---

### Task 6: Frontend — thumbs up/down + session UUID

**Files:**
- Modify: `frontend/src/types.ts`
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.css`

**Interfaces:**
- Consumes: the `AgentRunRead` response shape from Task 4 (now includes `recommendations: RecommendationRead[]`), the `POST /feedback` contract from Task 5 (`{recommendation_id, session_uuid, verdict}` → `FeedbackRead`).

- [ ] **Step 1: Add the new types**

In `frontend/src/types.ts`, add after the existing `ToolLogRead` interface:

```typescript
export interface RecommendationFeatures {
  cosine_sim: number
  tag_match_count: number
  budget_delta: number | null
  region_match: boolean
}

export interface RecommendationRead {
  id: number
  destination_id: string
  destination_name: string
  country: string
  rank_position: number
  score: number
  features: RecommendationFeatures
  created_at: string
}

export type FeedbackVerdict = 1 | -1

export interface FeedbackRead {
  id: number
  recommendation_id: number
  session_uuid: string
  verdict: FeedbackVerdict
  channel: string
  created_at: string
}
```

Then update `AgentRunRead` to add the new field:

```typescript
export interface AgentRunRead {
  id: number
  user_id: number
  prompt: string
  response: string
  status: string
  created_at: string
  tool_logs: ToolLogRead[]
  recommendations: RecommendationRead[]
}
```

- [ ] **Step 2: Add the `submitFeedback` API call**

In `frontend/src/lib/api.ts`, add `FeedbackRead` and `FeedbackVerdict` to the type-only import at the top:

```typescript
import type {
  AgentRunRead,
  FeedbackRead,
  FeedbackVerdict,
  PlannerRequest,
  TokenResponse,
  UserRead,
} from '../types'
```

Then add this function, right after `createAgentRun` and before the final `export { ApiError, API_BASE_URL }` line:

```typescript
export async function submitFeedback(payload: {
  recommendation_id: number
  session_uuid: string
  verdict: FeedbackVerdict
}): Promise<FeedbackRead> {
  return request<FeedbackRead>('/feedback', {
    method: 'POST',
    body: payload,
  })
}
```

- [ ] **Step 3: Add session UUID generation and the feedback panel to `App.tsx`**

In `frontend/src/App.tsx`, update the type-only import from `./types` to include the new types:

```typescript
import type { AgentRunRead, AuthMode, FeedbackVerdict, SessionState } from './types'
```

And update the value import from `./lib/api` to include `submitFeedback`:

```typescript
import {
  ApiError,
  createAgentRun,
  fetchCurrentUser,
  login,
  signup,
  submitFeedback,
} from './lib/api'
```

Add this constant near the top of the file, alongside `APP_ROUTE`/`LOGIN_ROUTE`/`SIGNUP_ROUTE`:

```typescript
const FEEDBACK_SESSION_STORAGE_KEY = 'smart-travel-feedback-session-uuid'

function getOrCreateFeedbackSessionUuid(): string {
  const existing = window.localStorage.getItem(FEEDBACK_SESSION_STORAGE_KEY)
  if (existing) {
    return existing
  }
  const created = crypto.randomUUID()
  window.localStorage.setItem(FEEDBACK_SESSION_STORAGE_KEY, created)
  return created
}
```

Inside the `App()` function, add these three new pieces of state right after the existing `plannerPending` state line:

```typescript
  const [feedbackSessionUuid] = useState(getOrCreateFeedbackSessionUuid)
  const [feedbackByRecommendation, setFeedbackByRecommendation] = useState<
    Record<number, FeedbackVerdict>
  >({})
  const [feedbackError, setFeedbackError] = useState('')
```

Add this handler function right after `handlePlanSubmit`:

```typescript
  async function handleFeedback(recommendationId: number, verdict: FeedbackVerdict) {
    setFeedbackError('')

    try {
      await submitFeedback({
        recommendation_id: recommendationId,
        session_uuid: feedbackSessionUuid,
        verdict,
      })
      setFeedbackByRecommendation((previous) => ({
        ...previous,
        [recommendationId]: verdict,
      }))
    } catch (error) {
      setFeedbackError(
        error instanceof ApiError ? error.message : 'Feedback could not be submitted.',
      )
    }
  }
```

Finally, add a new `<article>` panel inside the existing `<section className="results-grid">`, right after the closing `</article>` of `result-panel` and before the opening `<article className="panel logs-panel">`:

```tsx
        <article className="panel recommendations-panel">
          <div className="panel-heading">
            <div>
              <p className="panel-label">Recommendations</p>
              <h2>Rate the ranked slate</h2>
            </div>
            <span className="status-pill">
              {result ? `${result.recommendations.length} destinations` : 'No slate yet'}
            </span>
          </div>

          {feedbackError ? <p className="error-text">{feedbackError}</p> : null}

          {result?.recommendations.length ? (
            <div className="logs-list">
              {result.recommendations.map((recommendation) => {
                const activeVerdict = feedbackByRecommendation[recommendation.id]
                return (
                  <article key={recommendation.id} className="log-card">
                    <div className="log-header">
                      <strong>
                        #{recommendation.rank_position} {recommendation.destination_name}, {recommendation.country}
                      </strong>
                      <span className="log-status">{recommendation.score.toFixed(4)}</span>
                    </div>
                    <div className="feedback-actions">
                      <button
                        type="button"
                        className={
                          activeVerdict === 1
                            ? 'feedback-button feedback-button-active-up'
                            : 'feedback-button'
                        }
                        aria-pressed={activeVerdict === 1}
                        onClick={() => handleFeedback(recommendation.id, 1)}
                      >
                        Good match
                      </button>
                      <button
                        type="button"
                        className={
                          activeVerdict === -1
                            ? 'feedback-button feedback-button-active-down'
                            : 'feedback-button'
                        }
                        aria-pressed={activeVerdict === -1}
                        onClick={() => handleFeedback(recommendation.id, -1)}
                      >
                        Not a fit
                      </button>
                    </div>
                  </article>
                )
              })}
            </div>
          ) : (
            <p className="empty-state">
              Recommended destinations will appear here after a planner run, so
              you can rate each ranked result.
            </p>
          )}
        </article>

```

- [ ] **Step 4: Add feedback button styling**

In `frontend/src/App.css`, add this block right after the existing `.log-status { ... }` rule:

```css
.feedback-actions {
  display: flex;
  gap: 0.6rem;
  margin-top: 0.85rem;
}

.feedback-button {
  border: 1px solid rgba(32, 51, 55, 0.18);
  background: transparent;
  color: var(--ink);
  border-radius: 999px;
  padding: 0.4rem 0.9rem;
  font-size: 0.85rem;
  cursor: pointer;
}

.feedback-button-active-up {
  background: rgba(70, 118, 121, 0.18);
  border-color: rgba(70, 118, 121, 0.45);
}

.feedback-button-active-down {
  background: rgba(190, 90, 70, 0.16);
  border-color: rgba(190, 90, 70, 0.45);
}
```

- [ ] **Step 5: Verify the frontend builds and type-checks**

Run (from `frontend/`):

```bash
npm run build
```

Expected: build completes successfully (Vite + `tsc` type-check passes) with no TypeScript errors about `RecommendationRead`, `FeedbackVerdict`, or the new `App.tsx` state/handlers.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/types.ts frontend/src/lib/api.ts frontend/src/App.tsx frontend/src/App.css
git commit -m "feat(frontend): add recommendation slate display with thumbs up/down feedback"
```

---

### Task 7: Documentation

**Files:**
- Modify: `backend/README.md`
- Modify: `README.md` (repo root)

**Interfaces:** none — documentation only.

- [ ] **Step 1: Add the feedback data model section to `backend/README.md`**

Find the existing `## ML Feedback Schema (`recommendations`, `feedback`, `tag_definitions`)` section (added in the 2026-07-04 session, when these tables were schema-only) and add a new subsection directly after it, before the next `##` heading:

```markdown
### Feedback Data Model (now wired up)

As of this session, `recommendations` and `feedback` are no longer schema-only:

- **`recommendations`**: one row per candidate in the ranked slate the pre-filter+cosine node
  returned for a given `agent_runs` row - not just the top result. `features` is a JSONB snapshot
  captured once, at recommend time (`cosine_sim`, `tag_match_count`, `budget_delta`,
  `region_match`) - it is never recomputed later, so it stays an honest record of what the model
  saw at decision time even if the underlying destination's data changes afterward.
  Persisted in `app/services/agent_runs.py::create_agent_run`, right after the `agent_runs` row is
  committed (its `id` is the FK target, so persistence can only happen after that commit) - see
  `app/services/recommendation_persistence.py::persist_recommendation_slate`. A persistence
  failure is logged as its own `tool_logs` entry (`recommendation_persistence`, `status="failed"`)
  and does not fail the user-facing response, matching the existing Discord-webhook-delivery
  pattern in the same file.
- **`feedback`**: anonymous thumbs up/down, one row per `(recommendation_id, session_uuid)` -
  enforced by a real Postgres unique constraint (`uq_feedback_recommendation_session`, added in
  the `b1f4a9d3e7c2` migration), not just application-level deduplication. Re-submitting the same
  `(recommendation_id, session_uuid)` pair updates the existing row's `verdict` via a Postgres
  `INSERT ... ON CONFLICT ... DO UPDATE` upsert (`app/services/feedback.py::submit_feedback`)
  rather than creating a duplicate. `channel` (plain string, default `"web"`) exists so a future
  non-web feedback source (a CLI, a Discord reaction) can reuse this table without a migration.
- **`POST /feedback`** takes `{recommendation_id, session_uuid, verdict}` (`verdict` is `+1` or
  `-1`) and requires **no auth** - `session_uuid` is a random UUID the frontend generates once and
  stores in `localStorage`, never tied to a `users` row. This is the only no-PII, no-auth mutation
  endpoint in the API by design.
- **Slate → training-row mapping**: each `recommendations` row is a candidate training example -
  its `features` JSONB is the feature vector (X), and the associated `feedback.verdict` (when
  present) is the label (y). Rows with no matching `feedback` row are unlabeled candidates -
  useful for future negative sampling or exposure-weighting, not yet consumed by anything.
  Nothing currently trains on this table; populating it is the whole point of this session's work,
  actually wiring up a learning-to-rank model on top of it is still future work.
```

- [ ] **Step 2: Update the root `README.md`'s "Persistence" section**

Replace:

```markdown
## Persistence

The system uses one Postgres database for:

- users
- agent runs
- tool logs
- destination embeddings

### Current Tables

- `users`
- `agent_runs`
- `tool_logs`
- `destination_documents`

At minimum, persisted data includes:

- who asked
- what they asked
- what the agent answered
- which tools fired
- when the run happened
```

with:

```markdown
## Persistence

The system uses one Postgres database for:

- users
- agent runs
- tool logs
- destination embeddings
- recommendation slates + anonymous thumbs up/down feedback

### Current Tables

- `users`
- `agent_runs`
- `tool_logs`
- `destination_documents`
- `recommendations` - the full ranked slate per run, with a feature snapshot per candidate
- `feedback` - anonymous, session-uuid-keyed thumbs up/down on a recommendation

At minimum, persisted data includes:

- who asked
- what they asked
- what the agent answered
- which tools fired
- when the run happened
- which destinations were recommended, in what order, with what score
- whether an anonymous session rated a given recommendation useful
```

- [ ] **Step 3: Commit**

```bash
git add backend/README.md README.md
git commit -m "docs: document the feedback data model and slate persistence"
```

---

## Self-Review Notes

- Spec coverage: full-slate persistence (Task 3+4), `agent_run_id`/`destination_id`/`rank_position`/`score`/`features` exactly as specified (Task 3), features captured at recommend time and never recomputed (Task 4's non-fatal try/except wraps the existing dict from graph state, untouched), `POST /feedback` with `{recommendation_id, session_uuid, verdict}` (Task 5), idempotent on `(recommendation_id, session_uuid)` via a real DB constraint + upsert (Tasks 1 and 5), `channel` column defaulting to `"web"` (Task 1), frontend thumbs up/down + session_uuid in localStorage (Task 6), README slate→training-row mapping (Task 7) — all covered.
- No pytest introduced; every verification is a `uv run python -` scratch script against the real dev DB, each cleaning up its own scratch rows, per `CLAUDE.md`'s "no existing pattern to extend" guidance.
- Type consistency check: `RecommendationRead` (Task 2) fields match exactly what Task 3's `get_recommendations_for_agent_run` constructs, what Task 4's `AgentRunRead` embeds, and what Task 6's TypeScript interface mirrors. `FeedbackCreate`/`FeedbackRead` (Task 2) match what Task 5's route/service use and what Task 6's `submitFeedback` sends/expects.
