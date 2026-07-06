"""Coverage for app/services/clustering.py: membership thresholding and the
two DB tag write-back paths (cluster_id keys during `cluster`, tag_name keys
during `apply-tags`). Does not exercise run_clustering() itself (a real
UMAP+HDBSCAN fit) - too slow/flaky for a unit test and not what this audit
item asked for; ClusteringRun objects are constructed directly instead.
"""

import numpy as np
import pytest
from sqlalchemy import select

from app.db.models.destination import Destination
from app.db.models.tag_definition import TagDefinition
from app.services.clustering import (
    ClusteringRun,
    DestinationVector,
    apply_approved_tag_names,
    threshold_membership_to_tags,
    write_cluster_id_tags_to_db,
)


def test_threshold_membership_to_tags_excludes_weights_at_or_below_threshold():
    membership_row = np.array([0.9, 0.5, 0.05, 0.51])
    tags = threshold_membership_to_tags(membership_row, threshold=0.5)
    # index 1 (0.5) is excluded - strictly greater than threshold, not >=.
    assert tags == {"0": 0.9, "3": pytest.approx(0.51)}


def test_threshold_membership_to_tags_empty_when_nothing_clears_threshold():
    membership_row = np.array([0.1, 0.2, 0.3])
    tags = threshold_membership_to_tags(membership_row, threshold=0.5)
    assert tags == {}


def _make_clustering_run(destinations: list[Destination]) -> ClusteringRun:
    vectors = [
        DestinationVector(
            id=str(destination.id),
            name=destination.name,
            country=destination.country,
            region=destination.region,
            budget_level=destination.budget_level,
            embedding=np.asarray(destination.embedding, dtype=np.float64),
        )
        for destination in destinations
    ]
    # 3 destinations, 2 clusters: dest 0 -> cluster 0 only, dest 1 -> cluster
    # 1 only, dest 2 -> mixed membership in both (soft-clustering case).
    membership = np.array(
        [
            [0.9, 0.1],
            [0.05, 0.95],
            [0.6, 0.4],
        ]
    )
    labels = np.array([0, 1, 0])
    return ClusteringRun(
        vectors=vectors,
        umap_embedding=np.zeros((3, 2)),
        labels=labels,
        membership=membership,
        n_clusters=2,
        reducer=None,  # type: ignore[arg-type]
        clusterer=None,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_write_cluster_id_tags_to_db_writes_thresholded_weights(
    db_session, seeded_destinations
):
    subjects = seeded_destinations[:3]
    run = _make_clustering_run(subjects)

    await write_cluster_id_tags_to_db(db_session, run, membership_threshold=0.5)

    rows = (
        await db_session.execute(
            select(Destination.id, Destination.tags).where(
                Destination.id.in_([d.id for d in subjects])
            )
        )
    ).all()
    tags_by_id = {str(dest_id): tags for dest_id, tags in rows}

    assert tags_by_id[str(subjects[0].id)] == {"0": pytest.approx(0.9)}
    assert tags_by_id[str(subjects[1].id)] == {"1": pytest.approx(0.95)}
    # Destination 2 has membership 0.6 in cluster 0 and 0.4 in cluster 1 -
    # only cluster 0 clears the 0.5 threshold.
    assert tags_by_id[str(subjects[2].id)] == {"0": pytest.approx(0.6)}


@pytest.mark.asyncio(loop_scope="session")
async def test_apply_approved_tag_names_rewrites_cluster_id_keys_to_tag_names(
    db_session, seeded_destinations
):
    subjects = seeded_destinations[:3]
    db_session.add(
        TagDefinition(
            cluster_id=0,
            tag_name="Adventure",
            description="test fixture cluster",
            quality_metrics={},
        )
    )
    # Deliberately no TagDefinition for cluster 1 - simulates an
    # un-reviewed/un-named cluster that apply-tags must skip, not crash on.
    await db_session.commit()

    membership_dump = {
        "membership_threshold": 0.5,
        "n_clusters": 2,
        "clusters": {
            "0": [
                {"id": str(subjects[0].id), "membership": 0.9},
                {"id": str(subjects[2].id), "membership": 0.6},
            ],
            "1": [
                {"id": str(subjects[1].id), "membership": 0.95},
            ],
        },
    }

    result = await apply_approved_tag_names(db_session, membership_dump=membership_dump)

    rows = (
        await db_session.execute(
            select(Destination.id, Destination.tags).where(
                Destination.id.in_([d.id for d in subjects])
            )
        )
    ).all()
    tags_by_id = {str(dest_id): tags for dest_id, tags in rows}

    assert tags_by_id[str(subjects[0].id)] == {"Adventure": pytest.approx(0.9)}
    assert tags_by_id[str(subjects[2].id)] == {"Adventure": pytest.approx(0.6)}
    # Cluster 1 has no approved tag_definitions row - destination 1 gets no tags.
    assert tags_by_id[str(subjects[1].id)] == {}

    assert result["clusters_applied"] == [0]
    assert result["clusters_missing_names"] == [1]
    assert result["destinations_updated"] == 3
