"""Offline training pipeline for the destination ranker.

Shares app.services.ranker.FEATURE_NAMES/feature_vector() with the serve path
so a trained model always sees features encoded exactly the way they were
during training.
"""

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

from app.services.ranker import FEATURE_NAMES, feature_vector

# Noise added to the heuristic's raw relevance score before bucketing into
# graded labels, so the bootstrap labels aren't a purely deterministic
# function of the same four features the ranker is trained on. See
# heuristic_relevance_label()'s docstring for the full formula.
LABEL_NOISE_SIGMA = 0.15

LABEL_FORMULA_DESCRIPTION = (
    "raw = 3.0*cosine_sim + 0.5*min(tag_match_count, 2) + 0.5*region_match "
    "- 0.4*abs(budget_delta or 0), then raw += N(0, {sigma}) noise, then "
    "bucketed into 4 grades (0-3) via quantiles over the full bootstrap "
    "dataset."
).format(sigma=LABEL_NOISE_SIGMA)


def heuristic_relevance_label(features: dict[str, Any], rng: np.random.Generator) -> float:
    """COLD-START PRIOR - a synthetic relevance score, not real user feedback.

    Relevance rises with cosine_sim, tag_match_count (capped at 2, so no
    single feature dominates), and region_match; falls with |budget_delta|.
    Gaussian noise (sigma=LABEL_NOISE_SIGMA) is added so the ranker has to
    generalize rather than exactly re-deriving this formula from the same
    four features it trains on. Returns the raw noisy score - callers bucket
    it into integer grades across the whole dataset (bucket_into_grades()),
    not per-row, so the grade distribution stays balanced regardless of this
    formula's exact scale.
    """
    budget_delta = features.get("budget_delta")
    budget_penalty = 0.4 * abs(float(budget_delta)) if budget_delta is not None else 0.0
    raw = (
        3.0 * float(features["cosine_sim"])
        + 0.5 * min(float(features["tag_match_count"]), 2.0)
        + 0.5 * (1.0 if features["region_match"] else 0.0)
        - budget_penalty
    )
    return raw + rng.normal(0.0, LABEL_NOISE_SIGMA)


def bucket_into_grades(raw_scores: pd.Series, *, n_grades: int = 4) -> pd.Series:
    """Buckets raw heuristic scores into 0..n_grades-1 graded relevance labels.

    Quantile-based (pd.qcut) rather than fixed thresholds, so the label
    distribution stays balanced no matter the raw score's scale.
    labels=False returns integer bin codes directly, robust to pd.qcut
    merging duplicate-valued bin edges (which would otherwise mismatch an
    explicit labels=[...] list).
    """
    codes = pd.qcut(raw_scores, q=n_grades, labels=False, duplicates="drop")
    return codes.astype(int)


def train_lgbm_ranker(
    dataset: pd.DataFrame,
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
    n_estimators: int = 200,
) -> tuple[LGBMRanker, dict[str, Any]]:
    """Trains an LGBMRanker with a group-aware (by qid) train/val split.

    dataset must have columns: qid, label, cosine_sim, tag_match_count,
    budget_delta, region_match. Splitting by qid (not by row) guarantees no
    query's candidates are split across train and val.
    """
    dataset = dataset.copy()
    # pandas upcasts a partially-missing numeric column to float64 + NaN,
    # whether the dataset came from an in-memory list of dicts or a CSV
    # round-trip. feature_vector() expects a real None for "no budget
    # constraint was requested" - normalize NaN back to None here, the one
    # place both the bootstrap and real-feedback callers pass through.
    dataset["budget_delta"] = (
        dataset["budget_delta"].astype(object).where(dataset["budget_delta"].notna(), None)
    )

    rng = np.random.default_rng(seed)
    # np.asarray(...) matters here: dataset["qid"].unique() can come back as
    # a pandas StringArray (not a plain ndarray) when qid is a string column,
    # and np.random.Generator.shuffle only guarantees correct in-place
    # shuffling (no duplicate entries) on a real ndarray.
    qids = np.asarray(dataset["qid"].unique())
    rng.shuffle(qids)
    n_val_qids = max(1, int(len(qids) * val_fraction))
    val_qids = set(qids[:n_val_qids])

    train_df = dataset[~dataset["qid"].isin(val_qids)].sort_values("qid").reset_index(drop=True)
    val_df = dataset[dataset["qid"].isin(val_qids)].sort_values("qid").reset_index(drop=True)

    feature_columns = ["cosine_sim", "tag_match_count", "budget_delta", "region_match"]
    X_train = np.array(
        [feature_vector(row) for row in train_df[feature_columns].to_dict("records")], dtype=float
    )
    y_train = train_df["label"].to_numpy(dtype=int)
    group_train = train_df.groupby("qid", sort=False).size().to_numpy()

    X_val = np.array(
        [feature_vector(row) for row in val_df[feature_columns].to_dict("records")], dtype=float
    )
    y_val = val_df["label"].to_numpy(dtype=int)
    group_val = val_df.groupby("qid", sort=False).size().to_numpy()

    model = LGBMRanker(
        objective="lambdarank",
        n_estimators=n_estimators,
        random_state=seed,
        verbosity=-1,
    )
    model.fit(
        X_train,
        y_train,
        group=group_train,
        eval_set=[(X_val, y_val)],
        eval_group=[group_val],
        eval_at=[3, 5],
    )

    eval_scores = model.evals_result_["valid_0"]
    metrics: dict[str, Any] = {
        "ndcg_at_3": float(eval_scores["ndcg@3"][-1]),
        "ndcg_at_5": float(eval_scores["ndcg@5"][-1]),
        "feature_importances": {
            name: float(importance)
            for name, importance in zip(FEATURE_NAMES, model.feature_importances_, strict=True)
        },
        "n_train_rows": int(len(train_df)),
        "n_val_rows": int(len(val_df)),
        "n_train_groups": int(len(group_train)),
        "n_val_groups": int(len(group_val)),
    }
    return model, metrics


def build_model_metadata(
    *,
    dataset_source: str,
    dataset_path: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    trained_at = datetime.now(timezone.utc)
    metadata: dict[str, Any] = {
        "model_version": f"{dataset_source}-{trained_at.strftime('%Y%m%d%H%M%S')}",
        "trained_at": trained_at.isoformat(),
        "dataset_source": dataset_source,
        "dataset_path": dataset_path,
        "feature_names": FEATURE_NAMES,
        **metrics,
    }
    if dataset_source == "bootstrap":
        metadata["warning"] = (
            "COLD-START PRIOR: trained on synthetic queries + a heuristic label "
            "function, not real user feedback. See backend/README.md's "
            "'Learning-to-Rank' section."
        )
    return metadata
