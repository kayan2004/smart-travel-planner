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
