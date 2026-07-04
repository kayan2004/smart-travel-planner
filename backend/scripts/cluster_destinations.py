"""Offline HDBSCAN soft-clustering CLI for the destinations corpus.

Three separable, resumable phases (see backend/README.md "Destination
Clustering" section for the full write-up):

    cluster      Fit UMAP + HDBSCAN, write weighted {cluster_id: weight}
                 tags to destinations.tags, write quality/stability
                 artifacts to artifacts/clustering/.
    name         Ask Claude to propose a tag_name + description per
                 cluster (using artifacts from `cluster`), upsert into
                 tag_definitions. Re-runnable without re-clustering.
    apply-tags   Rewrite destinations.tags from cluster_id keys to
                 tag_name keys, once naming has been reviewed/approved
                 in tag_definitions.

This is an offline, run-once(-per-corpus-change) script - not a graph node,
never called from the request path.
"""

import argparse
import asyncio
import sys
from pathlib import Path

import httpx

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings
from app.db.session import create_db_engine, create_session_factory
from app.services import clustering

ARTIFACTS_DIR = BACKEND_DIR / "artifacts" / "clustering"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster the destinations corpus offline.")
    subparsers = parser.add_subparsers(dest="phase", required=True)

    cluster_parser = subparsers.add_parser("cluster", help="Phase 1 + 3: fit and validate.")
    cluster_parser.add_argument("--min-cluster-size", type=int, default=7)
    cluster_parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="Defaults to HDBSCAN's own default (equal to min-cluster-size) when omitted.",
    )
    cluster_parser.add_argument("--membership-threshold", type=float, default=0.15)
    cluster_parser.add_argument("--umap-n-components", type=int, default=10)
    cluster_parser.add_argument("--umap-n-neighbors", type=int, default=15)
    cluster_parser.add_argument("--umap-min-dist", type=float, default=0.0)
    cluster_parser.add_argument("--random-state", type=int, default=42)
    cluster_parser.add_argument("--min-corpus-size", type=int, default=50)
    cluster_parser.add_argument(
        "--n-stability-runs",
        type=int,
        default=5,
        help="Number of additional UMAP/HDBSCAN re-fits (varying UMAP's random_state) used to "
        "compute pairwise Adjusted Rand Index for the stability report.",
    )
    cluster_parser.add_argument(
        "--skip-stability",
        action="store_true",
        help="Skip the stability re-run grid (faster iteration during tuning).",
    )
    cluster_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute everything and write artifacts, but do not write destinations.tags.",
    )

    name_parser = subparsers.add_parser("name", help="Phase 2: propose + persist cluster names.")
    name_parser.add_argument(
        "--top-n",
        type=int,
        default=8,
        help="Top-membership example destinations shown to Claude per cluster.",
    )

    subparsers.add_parser(
        "apply-tags",
        help="Rewrite destinations.tags from cluster_id keys to approved tag_name keys.",
    )

    return parser.parse_args()


async def _run_cluster_phase(args: argparse.Namespace) -> None:
    settings = get_settings()
    engine = create_db_engine(settings)
    session_factory = create_session_factory(engine)

    try:
        async with session_factory() as session:
            vectors = await clustering.load_embedded_destination_vectors(session)

            if len(vectors) < args.min_corpus_size:
                print(
                    f"Aborting: only {len(vectors)} destinations have embeddings "
                    f"(need >= {args.min_corpus_size}). Run "
                    "`uv run python scripts/ingest_destinations.py` (full corpus, no --limit) "
                    "first.",
                    file=sys.stderr,
                )
                sys.exit(1)

            print(f"Loaded {len(vectors)} embedded destinations.")

            try:
                run = clustering.run_clustering(
                    vectors,
                    umap_n_components=args.umap_n_components,
                    umap_n_neighbors=args.umap_n_neighbors,
                    umap_min_dist=args.umap_min_dist,
                    min_cluster_size=args.min_cluster_size,
                    min_samples=args.min_samples,
                    random_state=args.random_state,
                )
            except clustering.DegenerateClusteringError as exc:
                print(f"Aborting: {exc}", file=sys.stderr)
                sys.exit(1)

            quality_metrics = clustering.compute_quality_metrics(
                run, membership_threshold=args.membership_threshold
            )
            print(
                f"Found {run.n_clusters} clusters, noise ratio "
                f"{quality_metrics['noise_ratio']:.2%}, "
                f"silhouette={quality_metrics['silhouette_umap_space']}, "
                f"dbcv={quality_metrics['dbcv']}"
            )

            stability_report = None
            if not args.skip_stability:
                print(f"Running {args.n_stability_runs} stability re-fits ...")
                stability_report = clustering.run_stability_check(
                    vectors,
                    base_random_state=args.random_state,
                    n_runs=args.n_stability_runs,
                    umap_n_components=args.umap_n_components,
                    umap_n_neighbors=args.umap_n_neighbors,
                    umap_min_dist=args.umap_min_dist,
                    min_cluster_size=args.min_cluster_size,
                    min_samples=args.min_samples,
                )
                mean_ari = stability_report["mean_ari"]
                if mean_ari is None:
                    print("Stability: --n-stability-runs must be >= 2 to compute pairwise ARI.")
                else:
                    flag = " [UNSTABLE]" if stability_report["flagged_unstable"] else ""
                    print(f"Stability: mean pairwise ARI={mean_ari:.3f}{flag}")

            clustering.save_clustering_artifacts(
                run,
                membership_threshold=args.membership_threshold,
                quality_metrics=quality_metrics,
                stability_report=stability_report,
                artifacts_dir=ARTIFACTS_DIR,
            )
            print(f"Wrote artifacts to {ARTIFACTS_DIR}")

            if args.dry_run:
                print("Dry run: destinations.tags was not written.")
            else:
                await clustering.write_cluster_id_tags_to_db(
                    session, run, membership_threshold=args.membership_threshold
                )
                print("Wrote weighted {cluster_id: weight} tags to destinations.tags.")
    finally:
        await engine.dispose()


async def _run_name_phase(args: argparse.Namespace) -> None:
    settings = get_settings()
    engine = create_db_engine(settings)
    session_factory = create_session_factory(engine)
    http_client = httpx.AsyncClient(follow_redirects=True)

    try:
        membership_dump = clustering.load_membership_dump(ARTIFACTS_DIR)
        quality_report = clustering.load_quality_report(ARTIFACTS_DIR)

        async with session_factory() as session:
            results = await clustering.name_clusters(
                session,
                http_client,
                settings,
                membership_dump=membership_dump,
                quality_report=quality_report,
                top_n=args.top_n,
                artifacts_dir=ARTIFACTS_DIR,
            )

        print(f"Proposed and persisted names for {len(results)} clusters:")
        for result in sorted(results, key=lambda r: r.cluster_id):
            print(f"  cluster {result.cluster_id}: \"{result.tag_name}\" - {result.description}")
        print(
            "\nReview tag_definitions (tag_name/description) before running "
            "`apply-tags` to write these names onto destinations.tags."
        )
    finally:
        await http_client.aclose()
        await engine.dispose()


async def _run_apply_tags_phase() -> None:
    settings = get_settings()
    engine = create_db_engine(settings)
    session_factory = create_session_factory(engine)

    try:
        membership_dump = clustering.load_membership_dump(ARTIFACTS_DIR)

        async with session_factory() as session:
            summary = await clustering.apply_approved_tag_names(
                session, membership_dump=membership_dump
            )

        print(f"Updated destinations.tags for {summary['destinations_updated']} destinations.")
        print(f"Applied named clusters: {summary['clusters_applied']}")
        if summary["clusters_missing_names"]:
            print(
                f"Clusters with no tag_definitions entry yet (excluded from tags): "
                f"{summary['clusters_missing_names']}. Run `name` for these first."
            )
    finally:
        await engine.dispose()


async def main() -> None:
    args = _parse_args()
    if args.phase == "cluster":
        await _run_cluster_phase(args)
    elif args.phase == "name":
        await _run_name_phase(args)
    elif args.phase == "apply-tags":
        await _run_apply_tags_phase()


if __name__ == "__main__":
    asyncio.run(main())
