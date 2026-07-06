"""Coverage for app/services/ranker_training.py: the heuristic bootstrap
label function, quantile bucketing, and the group-aware train/val split
that must never leak a query's candidates across the split.
"""

import numpy as np
import pandas as pd
import pytest

from app.services.ranker_training import (
    bucket_into_grades,
    build_model_metadata,
    heuristic_relevance_label,
    train_lgbm_ranker,
)


class _ZeroNoiseRng:
    """A stand-in for np.random.Generator that always returns 0 noise, so
    heuristic_relevance_label's formula can be checked exactly.
    """

    def normal(self, loc: float, scale: float) -> float:
        return 0.0


def test_heuristic_label_matches_documented_formula():
    features = {
        "cosine_sim": 0.8,
        "tag_match_count": 1,
        "budget_delta": -1,
        "region_match": True,
    }
    # raw = 3.0*0.8 + 0.5*min(1,2) + 0.5*1 - 0.4*abs(-1) = 2.4 + 0.5 + 0.5 - 0.4 = 3.0
    result = heuristic_relevance_label(features, _ZeroNoiseRng())
    assert result == pytest.approx(3.0)


def test_heuristic_label_caps_tag_match_count_at_two():
    low = heuristic_relevance_label(
        {"cosine_sim": 0.5, "tag_match_count": 2, "budget_delta": None, "region_match": False},
        _ZeroNoiseRng(),
    )
    high = heuristic_relevance_label(
        {"cosine_sim": 0.5, "tag_match_count": 10, "budget_delta": None, "region_match": False},
        _ZeroNoiseRng(),
    )
    # tag_match_count is capped at 2 inside the formula - 2 vs 10 must score identically.
    assert low == pytest.approx(high)


def test_heuristic_label_treats_none_budget_delta_as_no_penalty():
    with_none = heuristic_relevance_label(
        {"cosine_sim": 0.5, "tag_match_count": 0, "budget_delta": None, "region_match": False},
        _ZeroNoiseRng(),
    )
    with_zero = heuristic_relevance_label(
        {"cosine_sim": 0.5, "tag_match_count": 0, "budget_delta": 0, "region_match": False},
        _ZeroNoiseRng(),
    )
    assert with_none == pytest.approx(with_zero)


def test_heuristic_label_adds_real_noise_from_a_real_generator():
    rng = np.random.default_rng(0)
    features = {"cosine_sim": 0.5, "tag_match_count": 0, "budget_delta": None, "region_match": False}
    samples = [heuristic_relevance_label(features, rng) for _ in range(50)]
    # Same input features every time - only the noise term should vary.
    assert len(set(samples)) > 1
    assert np.std(samples) == pytest.approx(0.15, abs=0.05)


def test_bucket_into_grades_returns_values_in_expected_range():
    rng = np.random.default_rng(0)
    raw_scores = pd.Series(rng.random(200))
    grades = bucket_into_grades(raw_scores)
    assert set(grades.unique()) <= {0, 1, 2, 3}
    # Quantile-based bucketing over 200 roughly-uniform values should be
    # close to evenly split across the 4 grades, not collapsed into one.
    counts = grades.value_counts()
    assert counts.min() >= 40


def test_bucket_into_grades_respects_custom_n_grades():
    rng = np.random.default_rng(1)
    raw_scores = pd.Series(rng.random(100))
    grades = bucket_into_grades(raw_scores, n_grades=2)
    assert set(grades.unique()) <= {0, 1}


def _make_dataset(n_groups: int, rows_per_group: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for group in range(n_groups):
        for _ in range(rows_per_group):
            features = {
                "cosine_sim": float(rng.random()),
                "tag_match_count": int(rng.integers(0, 3)),
                "budget_delta": None if rng.random() < 0.3 else int(rng.integers(-2, 3)),
                "region_match": bool(rng.random() < 0.5),
            }
            raw = heuristic_relevance_label(features, rng)
            rows.append({"qid": f"q{group}", "raw_score": raw, **features})
    dataset = pd.DataFrame(rows)
    dataset["label"] = bucket_into_grades(dataset["raw_score"])
    return dataset


def test_train_val_split_never_leaks_a_query_across_both_sides():
    dataset = _make_dataset(n_groups=20, rows_per_group=8, seed=42)
    model, metrics = train_lgbm_ranker(dataset, val_fraction=0.25, seed=1, n_estimators=10)

    # Re-derive which qids ended up in which split using the same seed/logic
    # train_lgbm_ranker uses internally, by checking group counts sum correctly.
    assert metrics["n_train_groups"] + metrics["n_val_groups"] == 20
    assert metrics["n_train_rows"] + metrics["n_val_rows"] == len(dataset)
    assert metrics["n_val_groups"] == 5  # 25% of 20 groups


def test_train_lgbm_ranker_reports_ndcg_and_feature_importances():
    dataset = _make_dataset(n_groups=20, rows_per_group=8, seed=7)
    model, metrics = train_lgbm_ranker(dataset, val_fraction=0.2, seed=1, n_estimators=10)

    assert 0.0 <= metrics["ndcg_at_3"] <= 1.0
    assert 0.0 <= metrics["ndcg_at_5"] <= 1.0
    assert set(metrics["feature_importances"].keys()) == {
        "cosine_sim",
        "tag_match_count",
        "budget_delta",
        "region_match",
    }
    predicted = model.predict([[0.9, 2, 0, 1.0]])
    assert len(predicted) == 1


def test_train_lgbm_ranker_handles_missing_budget_delta_without_crashing():
    """budget_delta=None rows upcast to NaN in a pandas DataFrame - this must
    not reach LightGBM as NaN (feature_vector() expects a real None).
    """
    dataset = _make_dataset(n_groups=10, rows_per_group=5, seed=3)
    assert dataset["budget_delta"].isna().any(), "fixture should include at least one None row"
    model, metrics = train_lgbm_ranker(dataset, val_fraction=0.2, seed=1, n_estimators=5)
    assert metrics["n_train_rows"] > 0


def test_build_model_metadata_flags_bootstrap_as_cold_start_prior():
    metadata = build_model_metadata(
        dataset_source="bootstrap",
        dataset_path="fake.csv",
        metrics={"ndcg_at_3": 0.9, "ndcg_at_5": 0.9, "feature_importances": {}},
    )
    assert "COLD-START PRIOR" in metadata["warning"]
    assert metadata["dataset_source"] == "bootstrap"


def test_build_model_metadata_real_feedback_has_no_warning():
    metadata = build_model_metadata(
        dataset_source="real_feedback",
        dataset_path="fake.csv",
        metrics={"ndcg_at_3": 0.9, "ndcg_at_5": 0.9, "feature_importances": {}},
    )
    assert "warning" not in metadata
