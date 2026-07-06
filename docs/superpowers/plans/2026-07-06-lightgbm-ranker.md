# LightGBM Learning-to-Rank Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task, inline in this session on a feature branch (no subagent dispatch — this repo lives under a OneDrive-synced folder, and a prior subagent-driven run landed a commit in the wrong checkout because of it; see the `pgvector-recommendation-node` plan's history). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a LightGBM `LGBMRanker` over the four-feature recommendation snapshot (`cosine_sim`, `tag_match_count`, `budget_delta`, `region_match`) and use it to re-rank the pgvector cosine-retrieved candidate slate at serve time, when a trained model exists and a config flag enables it — falling back to plain cosine order otherwise. Bootstrapped from a documented, clearly-labeled synthetic heuristic (the real `recommendations`/`feedback` tables are currently empty), with a real-feedback retrain path ready for when they're not.

**Architecture:** Cosine similarity + the SQL pre-filter (`app/services/destination_recommendations.py`) stay the retrieval mechanism — unchanged in what they retrieve, only in how the retrieved set is *ordered* before truncation to `payload.limit`. A new `app/services/ranker.py` owns the single feature-vector encoding (`feature_vector()`) shared by training and serving, plus a lazy `load_ranker_model()` loader that returns `None` when no model has been trained (the default state on a fresh checkout). A new `app/services/ranker_training.py` owns the heuristic bootstrap-label function and the group-aware LightGBM training routine, shared by both the bootstrap-train path and the real-feedback retrain path. A new offline CLI, `scripts/train_ranker.py`, has three subcommands (`bootstrap`, `train`, `retrain`) following the same argparse-subparser shape as `scripts/cluster_destinations.py`.

**Tech Stack:** LightGBM (`LGBMRanker`, `lambdarank` objective, NDCG eval), pandas, numpy, joblib (model persistence — same as the existing SVC classifier), SQLAlchemy 2.x async (asyncpg) for the bootstrap/retrain DB reads.

## Global Constraints

- **Pin lightgbm to an exact version** — already done (`backend/pyproject.toml`: `"lightgbm==4.6.0"`, `uv lock` re-run) as part of pre-planning verification; Task 0 just commits it.
- **Single shared feature-extraction function**: `app.services.ranker.feature_vector()` is the *only* place a features dict is turned into the ranker's input vector — both `app/services/destination_recommendations.py` (serve) and `app/services/ranker_training.py` (train) call it. Never hand-roll the vector elsewhere.
- **Feature order is fixed**: `FEATURE_NAMES = ["cosine_sim", "tag_match_count", "budget_delta", "region_match"]`, defined once in `ranker.py`.
- **No query leakage across train/val splits**: splitting is always by the query-group id (`qid` — a synthetic id in bootstrap, `agent_run_id` in real-feedback), never by row.
- **Cosine retrieval is never removed**: the SQL pre-filter + pgvector `<=>` cosine order (`app/services/destination_recommendations.py`) remains exactly as it is; the ranker, when enabled and loaded, only reorders the already-retrieved candidate set before truncation to `limit`. When disabled or no model exists, behavior is byte-for-byte identical to before this plan.
- **The persisted `score` field always means cosine similarity**, never the ranker's score — so a `recommendations.features` snapshot stays an honest training input regardless of which order the ranker later put candidates in. Only `rank_position` reflects the ranker's reordering.
- **Bootstrap data is quarantined from real feedback, in code and docs**: every artifact/file/log line produced from synthetic labels says "COLD-START PRIOR" or "bootstrap" explicitly; the retrain path only activates once real `feedback` rows exist and never silently mixes bootstrap rows into a real-feedback training set.
- **Async is not required for the offline scripts** (`scripts/train_ranker.py`'s `bootstrap`/`retrain` subcommands use `asyncio.run()` around DB reads because the DB session factory is async-only, not because the offline pipeline itself needs to be async) — the serve path (`destination_recommendations.py`) stays fully async, unchanged in that respect.
- **This project has no automated test suite and none should be introduced** (see `CLAUDE.md`'s "Known gaps"). Verification throughout this plan is manual: run the actual script/service against the real local dev Postgres (already running, 219 real destinations, real embeddings) and inspect real output, mirroring how prior plans in this repo (`pgvector-recommendation-node`, `recommendation-slate-persistence-and-feedback`) verified their work. Any scratch DB rows a verification step creates are deleted by that same step.
- **No Voyage API calls in the bootstrap script**: `VOYAGE_API_KEY` is not currently configured in `backend/.env` (checked during pre-planning), so Phase 1 cannot call `embed_texts()` for new synthetic query text. Instead, synthetic query embeddings are built by perturbing a *real* destination's real embedding by a small calibrated Gaussian noise (see Task 3) — this still runs the real SQL pre-filter + cosine re-rank + feature-snapshot code path against the real corpus (satisfying "generate synthetic slates from the real corpus", the option chosen when this ambiguity was raised), it just doesn't need a live embedding API call to do it.
- **Standing workflow preference**: at the end, `finishing-a-development-branch` will present its 4-option menu — this user's default answer is "merge to main locally" (option 1), though the menu is still presented per the skill's own rules.

---

## File Structure

**Create:**
- `backend/app/services/ranker.py` — `FEATURE_NAMES`, `feature_vector()`, `load_ranker_model()`, `rank_order()`. Imported by both the serve path and the training modules.
- `backend/app/services/ranker_training.py` — `LABEL_NOISE_SIGMA`, `LABEL_FORMULA_DESCRIPTION`, `heuristic_relevance_label()`, `bucket_into_grades()`, `train_lgbm_ranker()`, `build_model_metadata()`.
- `backend/scripts/train_ranker.py` — CLI with `bootstrap` / `train` / `retrain` subcommands.
- `backend/artifacts/ranker/` — generated by running the script: `bootstrap_dataset.csv`, `bootstrap_dataset_meta.json`, `model.joblib`, `model_metadata.json`, (later) `real_feedback_dataset.csv`.

**Modify:**
- `backend/app/services/destination_recommendations.py` — extract `_build_feature_snapshot()`, compute feature snapshots for the *full* candidate set before truncation, conditionally re-rank via the loaded model, then truncate to `payload.limit`.
- `backend/app/core/config.py` — add `ranker_enabled: bool = False`.
- `backend/.env.example` — add `RANKER_ENABLED=false` with a short comment.
- `backend/README.md` — new "Learning-to-Rank: Cold-Start Bootstrap Ranker" section (label formula, honest cold-start framing, NDCG/importance report, config flag, retrain path).

---

### Task 0: Branch + commit the pinned lightgbm dependency

**Files:**
- Modify (already changed on disk, needs committing): `backend/pyproject.toml`, `backend/uv.lock`

- [ ] **Step 1: Create the feature branch**

```bash
cd "C:/Users/Kayan/OneDrive/Desktop/projects/sef_projects/smart_travel_assistant"
git status --short
git checkout -b feat/lightgbm-ranker
```

Expected: only `backend/pyproject.toml` and `backend/uv.lock` show as modified (the `lightgbm==4.6.0` pin added during pre-planning verification); then confirmation you're on the new branch.

- [ ] **Step 2: Verify lightgbm still imports and trains after the pin**

```bash
cd backend
uv run python -c "from lightgbm import LGBMRanker; print(LGBMRanker)"
```

Expected: prints the class, no import error.

- [ ] **Step 3: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock
git commit -m "chore(deps): pin lightgbm==4.6.0 for the destination ranker"
```

---

### Task 1: Shared feature-vector module + serve-time integration behind a config flag

**Files:**
- Create: `backend/app/services/ranker.py`
- Modify: `backend/app/core/config.py`
- Modify: `backend/.env.example`
- Modify: `backend/app/services/destination_recommendations.py`

**Interfaces:**
- Produces: `feature_vector(features: dict) -> list[float]`, `FEATURE_NAMES: list[str]`, `load_ranker_model() -> Any | None`, `rank_order(model, feature_rows: list[list[float]]) -> list[int]` — all in `app/services/ranker.py`. `_build_feature_snapshot(destination, distance, payload) -> DestinationFeatureSnapshot` in `destination_recommendations.py`, used by Task 3's bootstrap script.
- Consumes: nothing new — `destination_recommendations.py` already has `Destination`, `DestinationFeatureSnapshot`, `DestinationRecommendationRequest` imported.

- [ ] **Step 1: Create `app/services/ranker.py`**

```python
"""Serve-time ranker: feature-vector encoding + model loading + reranking.

The feature vector produced here MUST match the column order used when
training (see app/services/ranker_training.py and scripts/train_ranker.py) -
this module is the single shared definition both sides import, so train and
serve can never drift apart.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

from joblib import load

MODEL_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[2] / "artifacts" / "ranker" / "model.joblib"
)

# Fixed column order for the ranker's feature vector. Both training
# (ranker_training.py) and serving (destination_recommendations.py) build
# vectors through feature_vector() below, never by hand, so this order is
# the only place it's defined.
FEATURE_NAMES: list[str] = ["cosine_sim", "tag_match_count", "budget_delta", "region_match"]


def feature_vector(features: dict[str, Any]) -> list[float]:
    """Encodes a DestinationFeatureSnapshot-shaped dict into the ranker's input order.

    budget_delta is nullable (no budget constraint was requested, or the
    destination has no budget_level) - encoded as 0.0, a neutral "no signal"
    value. region_match is a bool, encoded as 1.0/0.0.
    """
    budget_delta = features.get("budget_delta")
    return [
        float(features["cosine_sim"]),
        float(features["tag_match_count"]),
        float(budget_delta) if budget_delta is not None else 0.0,
        1.0 if features["region_match"] else 0.0,
    ]


@lru_cache(maxsize=1)
def load_ranker_model() -> Any | None:
    """Loads the trained LGBMRanker, or None if no model has been trained yet."""
    if not MODEL_ARTIFACT_PATH.exists():
        return None
    return load(MODEL_ARTIFACT_PATH)


def rank_order(model: Any, feature_rows: list[list[float]]) -> list[int]:
    """Returns indices into feature_rows, best-predicted-relevance first."""
    scores = model.predict(feature_rows)
    return sorted(range(len(feature_rows)), key=lambda index: scores[index], reverse=True)
```

- [ ] **Step 2: Add the config flag**

In `backend/app/core/config.py`, add this line right after `gemini_temperature: float = 0.2` (end of the Gemini settings block):

```python
    gemini_temperature: float = 0.2
    # Off by default: the shipped model (if any) is trained on a synthetic
    # cold-start bootstrap, not real feedback - see backend/README.md's
    # "Learning-to-Rank" section. When True AND artifacts/ranker/model.joblib
    # exists, recommend_destinations() re-ranks the cosine-retrieved slate
    # with it; otherwise cosine order is used, exactly as before.
    ranker_enabled: bool = False
```

- [ ] **Step 3: Add to `.env.example`**

Find the Gemini section in `backend/.env.example` and add after it:

```
# Learning-to-rank (optional, off by default - see backend/README.md's
# "Learning-to-Rank" section). Requires artifacts/ranker/model.joblib to
# exist (run scripts/train_ranker.py) - if it doesn't, this flag is ignored
# and cosine order is used regardless.
RANKER_ENABLED=false
```

- [ ] **Step 4: Refactor `destination_recommendations.py` — extract the feature snapshot builder and integrate the ranker**

Add this import near the top (alongside the existing `app.services.voyage_embeddings` import):

```python
from app.services.ranker import feature_vector, load_ranker_model, rank_order
```

Replace the tail of `recommend_destinations()` — from `ranked = rows[: payload.limit]` (line 44 of the current file) through the end of the function — with (note: 4-space indent, one level inside the function body, matching the rest of the file — not 8):

```python
    candidates = [
        (destination, distance, _build_feature_snapshot(destination, distance, payload))
        for destination, distance in rows
    ]

    if settings.ranker_enabled:
        ranker_model = load_ranker_model()
        if ranker_model is not None:
            feature_rows = [
                feature_vector(snapshot.model_dump()) for _, _, snapshot in candidates
            ]
            order = rank_order(ranker_model, feature_rows)
            candidates = [candidates[index] for index in order]

    ranked = candidates[: payload.limit]
    results = [
        DestinationRecommendationItem(
            destination_id=destination.id,
            destination=destination.name,
            country=destination.country,
            region=destination.region,
            budget_level=destination.budget_level,
            score=round(1.0 - distance, 4),
            rank_position=index + 1,
            features=snapshot,
        )
        for index, (destination, distance, snapshot) in enumerate(ranked)
    ]

    return DestinationRecommendationResponse(
        query_text=query_text,
        count=len(results),
        used_relaxed_constraints=used_relaxed_constraints,
        results=results,
    )
```

Then add this new function after `recommend_destinations()` (before `_fetch_ranked_candidates`):

```python
def _build_feature_snapshot(
    destination: Destination,
    distance: float,
    payload: DestinationRecommendationRequest,
) -> DestinationFeatureSnapshot:
    return DestinationFeatureSnapshot(
        cosine_sim=round(1.0 - distance, 4),
        tag_match_count=_count_matching_tags(
            destination.tags, payload.required_tags, payload.tag_weight_threshold
        ),
        budget_delta=_budget_delta(destination.budget_level, payload.budget_level),
        region_match=_region_match(destination.region, payload.region),
    )
```

Note: `score` is always `round(1.0 - distance, 4)` (cosine similarity), never the ranker's score — only the *order* of `candidates` (and therefore `rank_position`) changes when the ranker is active.

- [ ] **Step 5: Verify behavior is unchanged with the flag off (default state, no model file yet)**

```bash
cd backend
uv run python -c "
import asyncio
import httpx
from app.core.config import get_settings
from app.db.session import create_db_engine, create_session_factory
from app.schemas.recommendations import DestinationRecommendationRequest
from app.services.destination_recommendations import recommend_destinations
from app.services.ranker import load_ranker_model

async def main():
    settings = get_settings()
    assert settings.ranker_enabled is False, 'expected ranker_enabled default False'
    assert load_ranker_model() is None, 'expected no model artifact yet'
    engine = create_db_engine(settings)
    factory = create_session_factory(engine)
    async with factory() as session, httpx.AsyncClient() as client:
        # A destination-name query needs a real VOYAGE_API_KEY to embed; skip
        # the embedding call and go straight at the cosine-order guarantee by
        # checking _fetch_ranked_candidates directly instead.
        from app.services.destination_recommendations import _fetch_ranked_candidates
        payload = DestinationRecommendationRequest(query_text='placeholder', limit=5, min_candidates=5)
        # A zero vector still exercises real SQL + ordering machinery.
        rows = await _fetch_ranked_candidates(session, payload, [0.0]*1024, fetch_limit=5, apply_filters=False)
        distances = [d for _, d in rows]
        assert distances == sorted(distances), 'rows must stay in ascending-distance (cosine) order'
        print('OK - cosine order preserved, ranker inert when disabled/missing:', len(rows), 'rows')
    await engine.dispose()

asyncio.run(main())
"
```

Expected: `OK - cosine order preserved, ranker inert when disabled/missing: 5 rows` with no assertion errors.

- [ ] **Step 6: Commit**

```bash
git add app/services/ranker.py app/services/destination_recommendations.py app/core/config.py .env.example
git commit -m "feat(ranker): add feature-vector module and serve-time rerank hook behind a config flag"
```

---

### Task 2: Bootstrap label heuristic + shared training routine

**Files:**
- Create: `backend/app/services/ranker_training.py`

**Interfaces:**
- Consumes: `app.services.ranker.FEATURE_NAMES`, `feature_vector()`.
- Produces: `LABEL_NOISE_SIGMA: float`, `LABEL_FORMULA_DESCRIPTION: str`, `heuristic_relevance_label(features: dict, rng: np.random.Generator) -> float`, `bucket_into_grades(raw_scores: pd.Series, *, n_grades: int = 4) -> pd.Series`, `train_lgbm_ranker(dataset: pd.DataFrame, *, val_fraction: float = 0.2, seed: int = 42, n_estimators: int = 200) -> tuple[LGBMRanker, dict]`, `build_model_metadata(*, dataset_source: str, dataset_path: str, metrics: dict) -> dict` — all consumed by Task 4/6's CLI subcommands.

- [ ] **Step 1: Create `app/services/ranker_training.py`**

```python
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
```

- [ ] **Step 2: Verify the pure functions with an in-memory synthetic dataset (no DB needed)**

```bash
cd backend
uv run python -c "
import numpy as np
import pandas as pd
from app.services.ranker_training import (
    bucket_into_grades, heuristic_relevance_label, train_lgbm_ranker, build_model_metadata,
)

rng = np.random.default_rng(0)
rows = []
for qid in range(30):
    for _ in range(10):
        features = {
            'cosine_sim': float(rng.random()),
            'tag_match_count': int(rng.integers(0, 3)),
            'budget_delta': None if rng.random() < 0.3 else int(rng.integers(-2, 3)),
            'region_match': bool(rng.random() < 0.5),
        }
        raw = heuristic_relevance_label(features, rng)
        rows.append({'qid': f'q{qid}', 'raw_score': raw, **features})

df = pd.DataFrame(rows)
df['label'] = bucket_into_grades(df['raw_score'])
assert set(df['label'].unique()) <= {0, 1, 2, 3}

model, metrics = train_lgbm_ranker(df, val_fraction=0.2, seed=1, n_estimators=20)
assert metrics['n_train_groups'] + metrics['n_val_groups'] == 30
train_qids = set(df['qid'].unique()) - set(df['qid'].unique()[:6])  # sanity only
print('metrics:', metrics)

metadata = build_model_metadata(dataset_source='bootstrap', dataset_path='fake.csv', metrics=metrics)
assert 'COLD-START PRIOR' in metadata['warning']
print('metadata warning present: OK')
"
```

Expected: no `UserWarning`, prints a `metrics` dict with `ndcg_at_3`/`ndcg_at_5` in `[0, 1]`, `n_train_groups`/`n_val_groups` summing to 30, and `metadata warning present: OK`.

- [ ] **Step 3: Commit**

```bash
git add app/services/ranker_training.py
git commit -m "feat(ranker): add heuristic bootstrap label function and shared LGBMRanker training routine"
```

---

### Task 3: Bootstrap dataset generator (`scripts/train_ranker.py bootstrap`)

**Files:**
- Create: `backend/scripts/train_ranker.py`

**Interfaces:**
- Consumes: `app.services.destination_recommendations._build_feature_snapshot`, `_fetch_ranked_candidates` (both from Task 1); `app.services.ranker_training.heuristic_relevance_label`, `bucket_into_grades`, `LABEL_NOISE_SIGMA`, `LABEL_FORMULA_DESCRIPTION` (Task 2).
- Produces: `artifacts/ranker/bootstrap_dataset.csv` (columns: `qid, destination_id, cosine_sim, tag_match_count, budget_delta, region_match, raw_score, label`), `artifacts/ranker/bootstrap_dataset_meta.json`. Task 4's `train` subcommand reads the CSV by default.

- [ ] **Step 1: Create `scripts/train_ranker.py`**

```python
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
from sqlalchemy import select

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings
from app.db.session import create_db_engine, create_session_factory
from app.db.models.destination import Destination
from app.schemas.recommendations import DestinationRecommendationRequest
from app.services.destination_recommendations import (
    _build_feature_snapshot,
    _fetch_ranked_candidates,
)
from app.services.ranker_training import (
    LABEL_FORMULA_DESCRIPTION,
    LABEL_NOISE_SIGMA,
    bucket_into_grades,
    heuristic_relevance_label,
)

ARTIFACTS_DIR = BACKEND_DIR / "artifacts" / "ranker"
BOOTSTRAP_DATASET_PATH = ARTIFACTS_DIR / "bootstrap_dataset.csv"
BOOTSTRAP_META_PATH = ARTIFACTS_DIR / "bootstrap_dataset_meta.json"

# How far a synthetic query's embedding is perturbed from its seed
# destination's real embedding, in the same units as the embedding itself
# (Voyage embeddings here are unit-norm, per-dimension std ~0.03 - checked
# against the real destinations table during pre-planning). Calibrated so
# the mean cosine_sim between a synthetic query and its own seed destination
# lands around 0.90 - close but never a 1.0 exact self-match, which would
# over-represent perfect relevance in the bootstrap set. There is no
# VOYAGE_API_KEY configured in this environment, so synthetic queries are
# built this way instead of embedding new query text - see this plan's
# Global Constraints.
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
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "n_seed_queries": len(seeds),
        "n_rows": len(dataset),
        "query_perturbation_sigma": QUERY_PERTURBATION_SIGMA,
        "label_noise_sigma": LABEL_NOISE_SIGMA,
        "label_formula": LABEL_FORMULA_DESCRIPTION,
    }
    BOOTSTRAP_META_PATH.write_text(json.dumps(metadata, indent=2))

    print(f"Wrote {len(dataset)} rows across {dataset['qid'].nunique()} queries to {BOOTSTRAP_DATASET_PATH}")
    print(f"Label distribution:\n{dataset['label'].value_counts().sort_index()}")


def main() -> None:
    args = _parse_args()
    if args.phase == "bootstrap":
        asyncio.run(_run_bootstrap(args))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the bootstrap phase against the real dev DB**

```bash
cd backend
uv run python scripts/train_ranker.py bootstrap
```

Expected: prints `Generating 150 synthetic queries against the real destinations corpus...`, then `Wrote <N> rows across 150 queries to ...bootstrap_dataset.csv` (N should be close to `150 * 15 = 2250`, possibly a little less if some queries' relaxed fallback still returned under 15), then a `label` value-count breakdown across `0,1,2,3`.

- [ ] **Step 3: Inspect the output**

```bash
uv run python -c "
import pandas as pd
df = pd.read_csv('artifacts/ranker/bootstrap_dataset.csv')
print(df.shape)
print(df.head())
print('qid groups:', df['qid'].nunique())
print('label counts:', df['label'].value_counts().sort_index().to_dict())
print('cosine_sim range:', df['cosine_sim'].min(), df['cosine_sim'].max())
"
cat artifacts/ranker/bootstrap_dataset_meta.json
```

Expected: a real DataFrame with the 8 documented columns, ~150 query groups, a 4-way label spread that isn't wildly lopsided (quantile bucketing guarantees this), `cosine_sim` values spread below 1.0 (no exact self-match at 1.0 dominating), and the metadata JSON's `warning` field visibly saying `COLD-START PRIOR`.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_ranker.py artifacts/ranker/bootstrap_dataset.csv artifacts/ranker/bootstrap_dataset_meta.json
git commit -m "feat(ranker): add bootstrap dataset generator (Phase 1, cold-start prior)"
```

---

### Task 4: Training subcommand (`scripts/train_ranker.py train`)

**Files:**
- Modify: `backend/scripts/train_ranker.py`

**Interfaces:**
- Consumes: `app.services.ranker_training.train_lgbm_ranker`, `build_model_metadata` (Task 2).
- Produces: `artifacts/ranker/model.joblib` (loadable by `app.services.ranker.load_ranker_model()`), `artifacts/ranker/model_metadata.json`.

- [ ] **Step 1: Add the `train` subparser**

In `_parse_args()`, after the `bootstrap_parser` block and before `return parser.parse_args()`, add:

```python
    train_parser = subparsers.add_parser("train", help="Phase 2: train on a dataset CSV.")
    train_parser.add_argument("--dataset", type=Path, default=BOOTSTRAP_DATASET_PATH)
    train_parser.add_argument(
        "--dataset-source", choices=["bootstrap", "real_feedback"], default="bootstrap"
    )
    train_parser.add_argument("--val-fraction", type=float, default=0.2)
    train_parser.add_argument("--n-estimators", type=int, default=200)
    train_parser.add_argument("--seed", type=int, default=42)
```

- [ ] **Step 2: Add imports and the model artifact paths**

Add to the imports at the top:

```python
from joblib import dump

from app.services.ranker_training import (
    LABEL_FORMULA_DESCRIPTION,
    LABEL_NOISE_SIGMA,
    build_model_metadata,
    bucket_into_grades,
    heuristic_relevance_label,
    train_lgbm_ranker,
)
```

(This replaces the Task 3 import block for `ranker_training` — same module, more names.)

Add next to `BOOTSTRAP_META_PATH`:

```python
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
MODEL_METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
```

- [ ] **Step 3: Add `_run_train()`**

```python
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
```

- [ ] **Step 4: Wire it into `main()`**

```python
def main() -> None:
    args = _parse_args()
    if args.phase == "bootstrap":
        asyncio.run(_run_bootstrap(args))
    elif args.phase == "train":
        _run_train(args)
```

- [ ] **Step 5: Run training on the bootstrap dataset**

```bash
cd backend
uv run python scripts/train_ranker.py train
```

Expected: prints train/val row+group counts (val groups ≈ 20% of ~150 ≈ 30), `NDCG@3=`/`NDCG@5=` both in `[0, 1]` (a well-behaved bootstrap fit should land clearly above the ~0.5 random-order baseline, since the labels are a deterministic-plus-noise function of the same features), a feature-importances dict with all 4 `FEATURE_NAMES`, and `Saved model to .../artifacts/ranker/model.joblib`.

- [ ] **Step 6: Confirm the model loads via the serve-path loader**

```bash
uv run python -c "
from app.services.ranker import load_ranker_model, feature_vector
load_ranker_model.cache_clear()
model = load_ranker_model()
assert model is not None, 'expected a loaded model now that artifacts/ranker/model.joblib exists'
score = model.predict([feature_vector({'cosine_sim': 0.9, 'tag_match_count': 2, 'budget_delta': 0, 'region_match': True})])
print('OK - model loads and predicts:', score)
"
cat artifacts/ranker/model_metadata.json
```

Expected: `OK - model loads and predicts: [...]` with one float score, and the metadata JSON shows `"dataset_source": "bootstrap"` and the `COLD-START PRIOR` warning.

- [ ] **Step 7: Commit**

```bash
git add scripts/train_ranker.py artifacts/ranker/model.joblib artifacts/ranker/model_metadata.json
git commit -m "feat(ranker): add training subcommand (Phase 2), save model + NDCG/importance report"
```

---

### Task 5: End-to-end serve verification with a real trained model

**Files:** none (verification only — Task 1 already wrote the serve integration, Task 4 already produced a real model)

- [ ] **Step 1: Verify the ranker changes order relative to cosine, and that the fallback still works, via the real `recommend_destinations()` function**

The dev environment has no `VOYAGE_API_KEY` configured, so `recommend_destinations()`'s internal `embed_texts()` call is mocked here — exactly like this session's earlier `GeminiProvider` verification mocked the Gemini SDK call when live credentials weren't usable. Everything downstream of the embedding (SQL pre-filter, cosine order, feature snapshots, ranker rerank) is real.

```bash
cd backend
uv run python -c "
import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np
from sqlalchemy import select

from app.core.config import Settings, get_settings
from app.db.session import create_db_engine, create_session_factory
from app.db.models.destination import Destination
from app.schemas.recommendations import DestinationRecommendationRequest
from app.services import destination_recommendations
from app.services.ranker import load_ranker_model

async def main():
    base_settings = get_settings()
    engine = create_db_engine(base_settings)
    factory = create_session_factory(engine)

    async with factory() as session:
        seed = (
            await session.execute(
                select(Destination).where(Destination.embedding.is_not(None)).limit(1)
            )
        ).scalars().first()
        seed_embedding = np.array(seed.embedding, dtype=float)
        rng = np.random.default_rng(7)
        query_embedding = (seed_embedding + rng.normal(0.0, 0.015, size=seed_embedding.shape)).tolist()

        payload = DestinationRecommendationRequest(query_text='mocked query', limit=10, min_candidates=15)

        with patch.object(
            destination_recommendations, 'embed_texts', new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = [query_embedding]

            disabled_settings = Settings(ranker_enabled=False)
            async with httpx.AsyncClient() as client:
                cosine_response = await destination_recommendations.recommend_destinations(
                    session, client, disabled_settings, payload
                )

            assert load_ranker_model() is not None, 'expected Task 4 model to exist'
            enabled_settings = Settings(ranker_enabled=True)
            async with httpx.AsyncClient() as client:
                ranked_response = await destination_recommendations.recommend_destinations(
                    session, client, enabled_settings, payload
                )

    cosine_order = [item.destination_id for item in cosine_response.results]
    ranked_order = [item.destination_id for item in ranked_response.results]
    cosine_scores = {item.destination_id: item.score for item in cosine_response.results}
    ranked_scores = {item.destination_id: item.score for item in ranked_response.results}

    assert set(cosine_order) == set(ranked_order), 'ranker must not change the retrieved candidate set, only its order'
    assert cosine_scores == ranked_scores, 'score must always be cosine similarity, regardless of rerank'
    print('cosine order:  ', cosine_order)
    print('ranked order:  ', ranked_order)
    print('OK - same candidate set and scores; order may differ (ranker) or match (cosine ties)')

    await engine.dispose()

asyncio.run(main())
"
```

Expected: `OK - same candidate set and scores; order may differ (ranker) or match (cosine ties)` with both orderings printed as UUID lists (same set of UUIDs in both, `cosine_scores == ranked_scores` holds because `score` always means cosine similarity regardless of which order the ranker put things in).

- [ ] **Step 2: No commit needed** — this task only verifies Task 1 + Task 4's already-committed code.

---

### Task 6: Real-feedback retrain path (`scripts/train_ranker.py retrain`)

**Files:**
- Modify: `backend/scripts/train_ranker.py`

**Interfaces:**
- Consumes: `Recommendation`, `Feedback` ORM models; `train_lgbm_ranker`, `build_model_metadata` (Task 2).
- Produces: `artifacts/ranker/real_feedback_dataset.csv` (audit trail), overwrites `artifacts/ranker/model.joblib` + `model_metadata.json` with `dataset_source: "real_feedback"` once enough real feedback exists.

- [ ] **Step 1: Add the `retrain` subparser**

In `_parse_args()`, after the `train_parser` block:

```python
    retrain_parser = subparsers.add_parser(
        "retrain", help="Phase 4: build a real-feedback dataset and train on it."
    )
    retrain_parser.add_argument("--min-group-size", type=int, default=2)
    retrain_parser.add_argument("--val-fraction", type=float, default=0.2)
    retrain_parser.add_argument("--n-estimators", type=int, default=200)
    retrain_parser.add_argument("--seed", type=int, default=42)
```

- [ ] **Step 2: Add imports and the audit-trail path**

Add to imports:

```python
from app.db.models.feedback import Feedback
from app.db.models.recommendation import Recommendation
```

Add next to `MODEL_METADATA_PATH`:

```python
REAL_FEEDBACK_DATASET_PATH = ARTIFACTS_DIR / "real_feedback_dataset.csv"
```

- [ ] **Step 3: Add `_run_retrain()`**

```python
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
```

- [ ] **Step 4: Wire it into `main()`**

```python
def main() -> None:
    args = _parse_args()
    if args.phase == "bootstrap":
        asyncio.run(_run_bootstrap(args))
    elif args.phase == "train":
        _run_train(args)
    elif args.phase == "retrain":
        asyncio.run(_run_retrain(args))
```

- [ ] **Step 5: Verify the no-op guard against the real (currently empty) `feedback` table**

```bash
cd backend
uv run python scripts/train_ranker.py retrain
```

Expected: `No real feedback yet (feedback table has no rows joinable to recommendations). Nothing to retrain on - keeping the existing model in place.` — and confirm nothing changed:

```bash
git status --short artifacts/ranker/
```

Expected: no output (model.joblib/model_metadata.json untouched, no real_feedback_dataset.csv created).

- [ ] **Step 6: Verify the real path works, using temporary scratch DB rows, then clean up**

```bash
cd backend
uv run python -c "
import asyncio
import uuid

from sqlalchemy import delete, insert, select

from app.core.config import get_settings
from app.db.session import create_db_engine, create_session_factory
from app.db.models.agent_run import AgentRun
from app.db.models.recommendation import Recommendation
from app.db.models.feedback import Feedback
from app.db.models.destination import Destination

async def main():
    settings = get_settings()
    engine = create_db_engine(settings)
    factory = create_session_factory(engine)

    async with factory() as session:
        user_id = (await session.execute(select(AgentRun.user_id).limit(1))).scalars().first()
        destination_ids = list(
            (await session.execute(select(Destination.id).limit(5))).scalars().all()
        )

        agent_run_ids = []
        recommendation_ids = []
        for run_index in range(6):
            agent_run = AgentRun(
                user_id=user_id, prompt='scratch-verify', response='scratch-verify', status='completed'
            )
            session.add(agent_run)
            await session.flush()
            agent_run_ids.append(agent_run.id)

            for rank, destination_id in enumerate(destination_ids[:3], start=1):
                features = {
                    'cosine_sim': 0.9 - rank * 0.1,
                    'tag_match_count': rank % 3,
                    'budget_delta': 0,
                    'region_match': rank == 1,
                }
                result = await session.execute(
                    insert(Recommendation)
                    .values(agent_run_id=agent_run.id, destination_id=destination_id, rank_position=rank, score=features['cosine_sim'], features=features)
                    .returning(Recommendation.id)
                )
                recommendation_id = result.scalar_one()
                recommendation_ids.append(recommendation_id)
                verdict = 1 if rank == 1 else -1
                await session.execute(
                    insert(Feedback).values(recommendation_id=recommendation_id, session_uuid=uuid.uuid4(), verdict=verdict)
                )
        await session.commit()
        print('inserted scratch agent_run_ids:', agent_run_ids)

    await engine.dispose()

asyncio.run(main())
"
uv run python scripts/train_ranker.py retrain --min-group-size 2
cat artifacts/ranker/model_metadata.json
```

Expected: the insert script prints 6 scratch `agent_run_id`s; `retrain` prints `Wrote 18 real-feedback rows across 6 agent runs to ...real_feedback_dataset.csv`, then NDCG@3/@5 and `Retrained on real feedback and replaced .../model.joblib`; `model_metadata.json` now shows `"dataset_source": "real_feedback"` with no `COLD-START PRIOR` warning key.

Now clean up the scratch rows and restore the checked-in model to the honest bootstrap state (the committed `model.joblib` should reflect the documented cold-start bootstrap, not throwaway verification data). This must delete in FK dependency order (`feedback` -> `recommendations` -> `agent_runs`): a Core-style bulk `delete()` statement does **not** trigger SQLAlchemy's ORM `cascade="all, delete-orphan"` (that only fires for `session.delete()` on a loaded object), and neither FK has `ondelete=CASCADE` at the Postgres level either — a naive `DELETE FROM agent_runs WHERE ...` here would raise a `ForeignKeyViolation`.

```bash
uv run python -c "
import asyncio
from sqlalchemy import delete, select

from app.core.config import get_settings
from app.db.session import create_db_engine, create_session_factory
from app.db.models.agent_run import AgentRun
from app.db.models.recommendation import Recommendation
from app.db.models.feedback import Feedback

async def main():
    settings = get_settings()
    engine = create_db_engine(settings)
    factory = create_session_factory(engine)
    async with factory() as session:
        agent_run_ids = list(
            (
                await session.execute(select(AgentRun.id).where(AgentRun.prompt == 'scratch-verify'))
            ).scalars().all()
        )
        recommendation_ids = list(
            (
                await session.execute(
                    select(Recommendation.id).where(Recommendation.agent_run_id.in_(agent_run_ids))
                )
            ).scalars().all()
        )

        await session.execute(delete(Feedback).where(Feedback.recommendation_id.in_(recommendation_ids)))
        await session.execute(delete(Recommendation).where(Recommendation.agent_run_id.in_(agent_run_ids)))
        result = await session.execute(delete(AgentRun).where(AgentRun.id.in_(agent_run_ids)))
        print('deleted scratch agent_runs:', result.rowcount)
        await session.commit()
    await engine.dispose()

asyncio.run(main())
"
rm -f artifacts/ranker/real_feedback_dataset.csv
uv run python scripts/train_ranker.py train
```

Expected: `deleted scratch agent_runs: 6` (the `recommendations`/`feedback` rows cascade-delete via each model's `cascade="all, delete-orphan"` relationship), then the final `train` re-run restores `artifacts/ranker/model.joblib`/`model_metadata.json` to the bootstrap-sourced model that Task 4 produced.

- [ ] **Step 7: Commit**

```bash
git add scripts/train_ranker.py
git commit -m "feat(ranker): add real-feedback retrain path (Phase 4)"
```

(No `artifacts/ranker/*` changes to commit here — Step 6 deliberately restored the bootstrap model as the checked-in artifact.)

---

### Task 7: README documentation

**Files:**
- Modify: `backend/README.md`

- [ ] **Step 1: Add a "Learning-to-Rank: Cold-Start Bootstrap Ranker" section**

Add a new `##` section to `backend/README.md` (near the existing "Destination Clustering" / recommendation-pipeline sections):

````markdown
## Learning-to-Rank: Cold-Start Bootstrap Ranker

`recommend_destinations()` (`app/services/destination_recommendations.py`) retrieves candidates
via a structured SQL pre-filter + pgvector cosine re-rank, exactly as before. Optionally, when
`RANKER_ENABLED=true` **and** a trained model exists at `artifacts/ranker/model.joblib`, the
cosine-retrieved candidate set is re-ordered by a LightGBM `LGBMRanker` before truncation to
`limit` — cosine stays the retrieval mechanism; the ranker only reorders. `score` in the API
response always means cosine similarity, regardless of which order the ranker produced; only
`rank_position` reflects the ranker's order.

### Why bootstrap, and why that's clearly labeled

The `recommendations`/`feedback` tables (added to persist ranking training data - see "Feedback
Data Model" above) are **empty in this environment**: no real user has rated a recommendation yet.
So there is no real feedback to learn from. `scripts/train_ranker.py bootstrap` generates a
**COLD-START PRIOR** instead: synthetic query profiles, run through the real recommendation
pipeline against the real 219-destination corpus, labeled by a documented heuristic with added
noise - not real user preferences. Every artifact this produces (the dataset CSV's sibling
`bootstrap_dataset_meta.json`, the trained model's `model_metadata.json`) carries an explicit
`"COLD-START PRIOR"` warning string. Treat a bootstrap-trained ranker as a plumbing/demo artifact,
not a quality improvement over cosine order - the label formula (below) is itself a function of
the same four features the model trains on, so it mostly approximates that formula rather than
learning genuine user preference, and one feature (`tag_match_count`) is 0 for essentially every
real request today (`required_tags` is never populated by the LangGraph node - see
`app/agent/graph.py`'s `recommend_destinations_node`), so `cosine_sim` still dominates in
practice. `RANKER_ENABLED` therefore defaults to `false`.

Synthetic queries are built by perturbing a **real** destination's real Voyage embedding with
small Gaussian noise (`QUERY_PERTURBATION_SIGMA = 0.015`, calibrated so a synthetic query's
cosine similarity to its own seed destination lands around 0.90 - close, but never a 1.0 exact
self-match, which would have over-represented "perfect relevance" in the bootstrap set). This
was chosen over embedding new synthetic query *text* because this environment has no
`VOYAGE_API_KEY` configured; perturbing an existing real embedding still exercises the real SQL
pre-filter, cosine re-rank, and feature-snapshot code paths against the real corpus.

### Label heuristic (Phase 1)

```
raw = 3.0*cosine_sim + 0.5*min(tag_match_count, 2) + 0.5*region_match - 0.4*abs(budget_delta or 0)
raw += N(0, 0.15)   # LABEL_NOISE_SIGMA - keeps the ranker from exactly re-deriving this formula
label = quantile_bucket(raw, 4 grades)   # balanced 0-3 grades regardless of raw's scale
```

Defined once in `app/services/ranker_training.py`'s `heuristic_relevance_label()` /
`bucket_into_grades()` - the noise and the formula are the "documented heuristic" the Phase 1 spec
asked for.

### Training (Phase 2) and NDCG / feature importances

`scripts/train_ranker.py train` splits the dataset into train/val **by query group (`qid`)**, never
by row - a candidate is never split across train and val for the same query. Trains an
`LGBMRanker(objective="lambdarank")`, reports `ndcg@3`/`ndcg@5` on the held-out groups plus
per-feature importances, and saves `artifacts/ranker/model.joblib` +
`artifacts/ranker/model_metadata.json` (`model_version`, `trained_at`, `dataset_source`, metrics).

### Real-feedback retrain (Phase 4)

Once real `feedback` rows exist, `scripts/train_ranker.py retrain` joins `feedback.verdict` onto
`recommendations.features`, averages verdicts per recommendation (a recommendation can receive
feedback from more than one anonymous session), maps the mean verdict to a 3-grade label
(`round(mean_verdict) + 1` → 0/1/2), groups by `agent_run_id`, drops any agent_run with fewer than
`--min-group-size` (default 2) feedback-labeled candidates, and runs the exact same
`train_lgbm_ranker()` routine as the bootstrap path - the only differences are the label source and
`model_metadata.json`'s `dataset_source` switching from `"bootstrap"` to `"real_feedback"` (and the
`COLD-START PRIOR` warning disappearing). If there isn't at least 2 eligible agent_runs yet, it
prints a message and leaves the existing model in place rather than training on too little data.

### Config

`RANKER_ENABLED` (`backend/.env`, default `false`) - `recommend_destinations()` only re-ranks when
this is `true` **and** `artifacts/ranker/model.joblib` exists; otherwise cosine order is used,
identical to before this feature existed.
````

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document the learning-to-rank cold-start bootstrap ranker"
```

---

### Task 8: Final review and finish the branch

- [ ] **Step 1: Review the full diff against `main`**

```bash
cd "C:/Users/Kayan/OneDrive/Desktop/projects/sef_projects/smart_travel_assistant"
git diff main...feat/lightgbm-ranker --stat
```

Confirm the file list matches this plan's "File Structure" section (`app/services/ranker.py`, `app/services/ranker_training.py`, `scripts/train_ranker.py`, `app/services/destination_recommendations.py`, `app/core/config.py`, `.env.example`, `README.md`, `pyproject.toml`, `uv.lock`, plus `artifacts/ranker/*`) and no unrelated files.

- [ ] **Step 2: Re-run the Task 5 end-to-end verification once more** to confirm the full branch (not just Task 5's isolated commit) still behaves correctly - same script as Task 5 Step 1.

- [ ] **Step 3: Use `superpowers:finishing-a-development-branch`** to present the merge/PR/keep/discard menu and complete the branch per whichever option is chosen (this user's standing default is "merge to main locally").

---

