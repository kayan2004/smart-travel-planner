"""Offline CLI for the destination learning-to-rank model.

Three subcommands (see backend/README.md's "Learning-to-Rank" section for
the full write-up):

    bootstrap   Phase 1: generate a COLD-START PRIOR training set from
                synthetic query profiles run through the real
                recommend-and-feature-snapshot pipeline against the real
                destinations corpus, labeled via a documented heuristic
                (not real user feedback). Writes artifacts/ranker/
                bootstrap_dataset.csv + bootstrap_dataset_meta.json.
    train       Phase 2: trains an LGBMRanker on a dataset CSV (bootstrap or
                real-feedback), reports NDCG@3/@5 + feature importances, and
                saves artifacts/ranker/model.joblib + model_metadata.json.
    retrain     Phase 4: builds a training set from real feedback.verdict
                joined onto recommendations.features, then runs the same
                training routine as `train`. No-ops with a clear message if
                there isn't enough real feedback yet.

This is an offline, run-once(-per-corpus-or-feedback-change) script - not a
graph node, never called from the request path.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sqlalchemy import select

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings
from app.db.session import create_db_engine, create_session_factory
# Registers every ORM model's mapper before any query touches Recommendation
# or AgentRun - their relationship() targets are resolved by class name
# lazily, on first query, and every model on app.db.base.Base needs to have
# been imported somewhere first for that lookup to succeed (same pattern
# alembic/env.py uses for autogenerate).
from app.db.models import agent_run  # noqa: F401
from app.db.models import destination_document  # noqa: F401
from app.db.models import feedback  # noqa: F401
from app.db.models import recommendation  # noqa: F401
from app.db.models import tag_definition  # noqa: F401
from app.db.models import tool_log  # noqa: F401
from app.db.models import user  # noqa: F401
from app.db.models.destination import Destination
from app.db.models.feedback import Feedback
from app.db.models.recommendation import Recommendation
from app.schemas.recommendations import DestinationRecommendationRequest
from app.services.destination_recommendations import (
    _build_feature_snapshot,
    _fetch_ranked_candidates,
)
from app.services.ranker_training import (
    LABEL_FORMULA_DESCRIPTION,
    LABEL_NOISE_SIGMA,
    build_model_metadata,
    bucket_into_grades,
    heuristic_relevance_label,
    train_lgbm_ranker,
)

ARTIFACTS_DIR = BACKEND_DIR / "artifacts" / "ranker"
BOOTSTRAP_DATASET_PATH = ARTIFACTS_DIR / "bootstrap_dataset.csv"
BOOTSTRAP_META_PATH = ARTIFACTS_DIR / "bootstrap_dataset_meta.json"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
MODEL_METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
REAL_FEEDBACK_DATASET_PATH = ARTIFACTS_DIR / "real_feedback_dataset.csv"

# How far a synthetic query's embedding is perturbed from its seed
# destination's real embedding, in the same units as the embedding itself
# (Voyage embeddings here are unit-norm, per-dimension std ~0.03 - checked
# against the real destinations table during pre-planning). Calibrated so
# the mean cosine_sim between a synthetic query and its own seed destination
# lands around 0.90 - close but never a 1.0 exact self-match, which would
# over-represent perfect relevance in the bootstrap set. There is no
# VOYAGE_API_KEY configured in this environment, so synthetic queries are
# built this way instead of embedding new query text - see backend/README.md's
# "Learning-to-Rank" section.
QUERY_PERTURBATION_SIGMA = 0.015


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the destination learning-to-rank model.")
    subparsers = parser.add_subparsers(dest="phase", required=True)

    bootstrap_parser = subparsers.add_parser(
        "bootstrap", help="Phase 1: generate the cold-start bootstrap dataset."
    )
    bootstrap_parser.add_argument("--n-queries", type=int, default=150)
    bootstrap_parser.add_argument("--min-candidates", type=int, default=15)
    bootstrap_parser.add_argument("--seed", type=int, default=42)

    train_parser = subparsers.add_parser("train", help="Phase 2: train on a dataset CSV.")
    train_parser.add_argument("--dataset", type=Path, default=BOOTSTRAP_DATASET_PATH)
    train_parser.add_argument(
        "--dataset-source", choices=["bootstrap", "real_feedback"], default="bootstrap"
    )
    train_parser.add_argument("--val-fraction", type=float, default=0.2)
    train_parser.add_argument("--n-estimators", type=int, default=200)
    train_parser.add_argument("--seed", type=int, default=42)

    retrain_parser = subparsers.add_parser(
        "retrain", help="Phase 4: build a real-feedback dataset and train on it."
    )
    retrain_parser.add_argument("--min-group-size", type=int, default=2)
    retrain_parser.add_argument("--val-fraction", type=float, default=0.2)
    retrain_parser.add_argument("--n-estimators", type=int, default=200)
    retrain_parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def _synthetic_query_payload(
    seed: Destination, rng: np.random.Generator, min_candidates: int
) -> DestinationRecommendationRequest:
    budget_choice = rng.random()
    if budget_choice < 0.5:
        budget_level = seed.budget_level
    elif budget_choice < 0.75:
        budget_level = None
    else:
        budget_level = str(rng.choice(["low", "medium", "high"]))

    region = seed.region if rng.random() < 0.6 else None

    required_tags: list[str] = []
    eligible_tags = [key for key, weight in seed.tags.items() if float(weight) >= 0.5]
    if eligible_tags and rng.random() < 0.5:
        required_tags = [str(rng.choice(eligible_tags))]

    return DestinationRecommendationRequest(
        query_text=f"synthetic bootstrap query seeded from destination {seed.id}",
        budget_level=budget_level,
        region=region,
        required_tags=required_tags,
        limit=min_candidates,
        min_candidates=min_candidates,
    )


async def _run_bootstrap(args: argparse.Namespace) -> None:
    settings = get_settings()
    engine = create_db_engine(settings)
    session_factory = create_session_factory(engine)
    rng = np.random.default_rng(args.seed)

    rows: list[dict[str, Any]] = []

    async with session_factory() as session:
        seeds = list(
            (
                await session.execute(
                    select(Destination)
                    .where(Destination.deleted_at.is_(None))
                    .where(Destination.embedding.is_not(None))
                )
            )
            .scalars()
            .all()
        )

        if len(seeds) > args.n_queries:
            seed_indices = rng.choice(len(seeds), size=args.n_queries, replace=False)
            seeds = [seeds[i] for i in seed_indices]

        print(f"Generating {len(seeds)} synthetic queries against the real destinations corpus...")

        for seed in seeds:
            payload = _synthetic_query_payload(seed, rng, args.min_candidates)
            seed_embedding = np.array(seed.embedding, dtype=float)
            query_embedding = (
                seed_embedding + rng.normal(0.0, QUERY_PERTURBATION_SIGMA, size=seed_embedding.shape)
            ).tolist()

            candidates = await _fetch_ranked_candidates(
                session, payload, query_embedding, fetch_limit=args.min_candidates, apply_filters=True
            )
            if len(candidates) < args.min_candidates:
                candidates = await _fetch_ranked_candidates(
                    session, payload, query_embedding, fetch_limit=args.min_candidates, apply_filters=False
                )

            qid = f"synthetic-{seed.id}"
            for destination, distance in candidates:
                snapshot = _build_feature_snapshot(destination, distance, payload)
                features = snapshot.model_dump()
                raw_score = heuristic_relevance_label(features, rng)
                rows.append(
                    {
                        "qid": qid,
                        "destination_id": str(destination.id),
                        "cosine_sim": features["cosine_sim"],
                        "tag_match_count": features["tag_match_count"],
                        "budget_delta": features["budget_delta"],
                        "region_match": features["region_match"],
                        "raw_score": raw_score,
                    }
                )

    await engine.dispose()

    if not rows:
        print("No candidates were generated - is the destinations corpus seeded?")
        return

    dataset = pd.DataFrame(rows)
    dataset["label"] = bucket_into_grades(dataset["raw_score"])

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(BOOTSTRAP_DATASET_PATH, index=False)

    metadata = {
        "warning": (
            "COLD-START PRIOR: synthetic query profiles + a heuristic label "
            "function, not real user feedback. See backend/README.md's "
            "'Learning-to-Rank' section before treating this as ground truth."
        ),
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "n_seed_queries": len(seeds),
        "n_rows": len(dataset),
        "query_perturbation_sigma": QUERY_PERTURBATION_SIGMA,
        "label_noise_sigma": LABEL_NOISE_SIGMA,
        "label_formula": LABEL_FORMULA_DESCRIPTION,
    }
    BOOTSTRAP_META_PATH.write_text(json.dumps(metadata, indent=2))

    print(f"Wrote {len(dataset)} rows across {dataset['qid'].nunique()} queries to {BOOTSTRAP_DATASET_PATH}")
    print(f"Label distribution:\n{dataset['label'].value_counts().sort_index()}")


def _run_train(args: argparse.Namespace) -> None:
    if not args.dataset.exists():
        print(f"Dataset not found at {args.dataset} - run the 'bootstrap' phase first.")
        return

    dataset = pd.read_csv(args.dataset)
    model, metrics = train_lgbm_ranker(
        dataset,
        val_fraction=args.val_fraction,
        seed=args.seed,
        n_estimators=args.n_estimators,
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    dump(model, MODEL_PATH)
    metadata = build_model_metadata(
        dataset_source=args.dataset_source,
        dataset_path=str(args.dataset),
        metrics=metrics,
    )
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    print(f"Trained on {metrics['n_train_rows']} rows / {metrics['n_train_groups']} groups")
    print(f"Validated on {metrics['n_val_rows']} rows / {metrics['n_val_groups']} groups")
    print(f"NDCG@3={metrics['ndcg_at_3']:.4f}  NDCG@5={metrics['ndcg_at_5']:.4f}")
    print(f"Feature importances: {metrics['feature_importances']}")
    print(f"Saved model to {MODEL_PATH}")


async def _run_retrain(args: argparse.Namespace) -> None:
    settings = get_settings()
    engine = create_db_engine(settings)
    session_factory = create_session_factory(engine)

    async with session_factory() as session:
        rows_result = (
            await session.execute(
                select(
                    Recommendation.agent_run_id,
                    Recommendation.id,
                    Recommendation.destination_id,
                    Recommendation.features,
                    Feedback.verdict,
                )
                .join(Feedback, Feedback.recommendation_id == Recommendation.id)
                .where(Recommendation.deleted_at.is_(None))
                .where(Feedback.deleted_at.is_(None))
            )
        ).all()

    await engine.dispose()

    if not rows_result:
        print(
            "No real feedback yet (feedback table has no rows joinable to "
            "recommendations). Nothing to retrain on - keeping the existing "
            "model in place."
        )
        return

    per_recommendation: dict[int, dict[str, Any]] = {}
    verdicts: dict[int, list[int]] = {}
    for agent_run_id, recommendation_id, destination_id, features, verdict in rows_result:
        per_recommendation[recommendation_id] = {
            "qid": str(agent_run_id),
            "destination_id": str(destination_id),
            "cosine_sim": features["cosine_sim"],
            "tag_match_count": features["tag_match_count"],
            "budget_delta": features["budget_delta"],
            "region_match": features["region_match"],
        }
        verdicts.setdefault(recommendation_id, []).append(verdict)

    rows: list[dict[str, Any]] = []
    for recommendation_id, row in per_recommendation.items():
        mean_verdict = sum(verdicts[recommendation_id]) / len(verdicts[recommendation_id])
        # verdict in {-1, +1} -> label in {0, 1, 2}. Coarser than the
        # bootstrap's 4-grade scale (real feedback is a single vote, not a
        # continuous heuristic score), but the same lambdarank objective
        # only needs relative ordering within a group to work.
        rows.append({**row, "label": round(mean_verdict) + 1})

    dataset = pd.DataFrame(rows)
    group_sizes = dataset.groupby("qid").size()
    eligible_qids = group_sizes[group_sizes >= args.min_group_size].index
    dataset = dataset[dataset["qid"].isin(eligible_qids)]

    if dataset.empty or dataset["qid"].nunique() < 2:
        print(
            f"Not enough real feedback yet: need at least 2 agent_runs with "
            f">= {args.min_group_size} feedback-labeled recommendations each. "
            "Keeping the existing model in place."
        )
        return

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(REAL_FEEDBACK_DATASET_PATH, index=False)
    print(
        f"Wrote {len(dataset)} real-feedback rows across {dataset['qid'].nunique()} "
        f"agent runs to {REAL_FEEDBACK_DATASET_PATH}"
    )

    model, metrics = train_lgbm_ranker(
        dataset,
        val_fraction=args.val_fraction,
        seed=args.seed,
        n_estimators=args.n_estimators,
    )
    dump(model, MODEL_PATH)
    metadata = build_model_metadata(
        dataset_source="real_feedback",
        dataset_path=str(REAL_FEEDBACK_DATASET_PATH),
        metrics=metrics,
    )
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    print(f"NDCG@3={metrics['ndcg_at_3']:.4f}  NDCG@5={metrics['ndcg_at_5']:.4f}")
    print(f"Retrained on real feedback and replaced {MODEL_PATH}")


def main() -> None:
    args = _parse_args()
    if args.phase == "bootstrap":
        asyncio.run(_run_bootstrap(args))
    elif args.phase == "train":
        _run_train(args)
    elif args.phase == "retrain":
        asyncio.run(_run_retrain(args))


if __name__ == "__main__":
    main()
