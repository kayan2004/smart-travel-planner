"""Priority 1 coverage: app/services/destination_recommendations.py."""

import httpx
import pytest

from app.core.config import Settings, VoyageSettings
from app.schemas.recommendations import DestinationRecommendationRequest
from app.services.destination_recommendations import recommend_destinations
from tests.conftest import mock_voyage_transport

# Not get_settings() - these tests must not depend on the real environment
# having a real VOYAGE_API_KEY. embed_texts() raises before ever reaching
# the mocked transport if settings.voyage.api_key is empty, and CI has no
# .env file at all - this passed locally only because the dev .env happens
# to have a real key, masking the bug until CI ran it for real.
#
# ranker_enabled=False here is deliberate, not the production default: these
# tests exercise the SQL pre-filter/relaxation and raw cosine-similarity
# ordering in isolation. Leaving the real ranker on would let it reorder
# results by its own (cosine_sim/tag_match_count/budget_delta/region_match)
# scoring and confound what each test is actually checking - the ranker's
# own behavior belongs in a dedicated test, not folded into these.
def _test_settings() -> Settings:
    return Settings(voyage=VoyageSettings(api_key="test-voyage-key"), ranker_enabled=False)


@pytest.mark.asyncio(loop_scope="session")
async def test_budget_ceiling_allows_lower_and_equal_levels(db_session, seeded_destinations):
    settings = _test_settings()
    transport = mock_voyage_transport()
    async with httpx.AsyncClient(transport=transport) as client:
        # Exactly 7 of the 10 seeded destinations have budget_level in
        # {low, medium, None} - min_candidates must not exceed that, or the
        # relax-fallback (correctly) kicks in and returns everything,
        # including "high" destinations, defeating this test's point.
        payload = DestinationRecommendationRequest(
            query_text="a medium-budget trip", budget_level="medium", limit=7, min_candidates=5
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
    settings = _test_settings()
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
    settings = _test_settings()
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
    settings = _test_settings()
    transport = mock_voyage_transport()
    async with httpx.AsyncClient(transport=transport) as client:
        # Exactly 4 of the 10 seeded destinations are in Asia -
        # min_candidates must not exceed that, or the relax-fallback
        # (correctly) drops the region filter and returns every region.
        payload = DestinationRecommendationRequest(
            query_text="somewhere in Asia", region="Asia", limit=4, min_candidates=4
        )
        response = await recommend_destinations(db_session, client, settings, payload)

    returned_regions = {item.region for item in response.results}
    assert returned_regions == {"Asia"}


@pytest.mark.asyncio(loop_scope="session")
async def test_results_are_ordered_by_cosine_similarity_descending(db_session, seeded_destinations):
    settings = _test_settings()
    # Use the exact embedding of one seeded destination as the "user profile"
    # so its cosine similarity to itself is the maximum possible (1.0),
    # guaranteeing a predictable top result.
    target = seeded_destinations[0]
    # pgvector deserializes Destination.embedding as numpy float32 scalars,
    # which json.dumps can't serialize directly (used by mock_voyage_transport's
    # handler) - cast each element to a native Python float first.
    transport = mock_voyage_transport(embedding=[float(x) for x in target.embedding])
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
    settings = _test_settings()
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
    settings = _test_settings()
    transport = mock_voyage_transport()
    async with httpx.AsyncClient(transport=transport) as client:
        # Exactly 3 of the 10 seeded destinations are in Asia with
        # budget_level in {low, medium, None} - min_candidates must not
        # exceed that, or the relax-fallback returns non-Asia destinations
        # too, whose (correctly) region_match=False would fail this test
        # for the wrong reason.
        payload = DestinationRecommendationRequest(
            query_text="a medium-budget trip to Asia",
            budget_level="medium",
            region="Asia",
            limit=3,
            min_candidates=3,
        )
        response = await recommend_destinations(db_session, client, settings, payload)

    for item in response.results:
        assert item.features.cosine_sim == item.score
        assert item.features.region_match is True
        if item.budget_level is not None:
            # budget_delta = BUDGET_ORDER[destination] - BUDGET_ORDER[requested]
            assert isinstance(item.features.budget_delta, int)
