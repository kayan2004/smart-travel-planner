## Running Tests

![CI](https://github.com/kayan2004/smart-travel-assistant/actions/workflows/ci.yml/badge.svg)

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

Coverage target is ~70% of `app/services` + `app/agent` - a target, not a 100%-or-fail gate, and this
first suite doesn't hit it: overall `app/services` + `app/agent` coverage is ~26%, but that number is
dragged down by four entirely offline, never-request-path modules (`clustering.py`,
`destination_ingestion.py`, `rag_ingestion.py`, `ranker_training.py` - run manually via `scripts/*.py`,
never by the live app) sitting at 0%. The six priority areas the spec named are covered well
(`destination_recommendations.py` 90%, `auth.py` 96%, both LLM providers 95%): this suite covers
highest-value request-path code first, deliberately not the offline pipelines, and doesn't chase the
70% number by testing code nothing actually exercises at request time.

CI (`.github/workflows/ci.yml`) runs the same suite against a `postgres`/`pgvector` service
container on every push/PR, plus `ruff check .` and `mypy app` as separate jobs. `mypy` is
deliberately scoped to ignore a documented list of pre-existing-error modules (see
`backend/pyproject.toml`'s `[tool.mypy]` section) rather than gating on issues this test suite
didn't introduce.

## ML Workflow

This backend now includes a complete travel-style classification workflow built from `data/travel_destinations_labeled.csv`.

### Labels

The six target classes are:

- `Adventure`
- `Relaxation`
- `Culture`
- `Budget`
- `Luxury`
- `Family`

The labeled CSV is treated as a curated assignment artifact. It preserves the original destination features and adds:

- `travel_style`
- `label_status`
- `label_notes`

### Features

Training excludes `destination`, `country`, and the label/audit columns. The model uses:

- Categorical: `region`, `budget_level`, `tourism_level`
- Binary numeric: `has_hiking`, `has_beach`
- Continuous numeric: `culture_score`, `luxury_score`, `family_friendly`, `nightlife_level`, `avg_temp_peak`

### Training

Run the notebook from `backend/notebook/ml.ipynb`. The notebook now contains the full flow:

- exploratory data analysis
- compares Logistic Regression, Random Forest, and SVC
- uses 5-fold stratified cross-validation
- tunes Random Forest with grid search
- selects the winner by macro F1
- saves the trained model with `joblib`

### Artifacts

Training outputs are written to `artifacts/ml/`:

- `results.csv`
- `classification_report.json`
- `model_reports.json`
- `model_metadata.json`
- `best_model.joblib`

### Inference

Use the self-contained `predict_travel_style()` helper inside the notebook with a single destination-shaped feature dictionary to get a predicted travel style and probabilities when supported by the model.

## API Skeleton

The backend now has an initial FastAPI skeleton with:

- typed settings in `app/core/config.py`
- lifespan-managed app state in `app/core/lifespan.py`
- async database engine/session wiring in `app/db/`
- a starter health route in `app/api/routes/health.py`

Run it from the `backend` directory with:

```powershell
uv run python main.py
```

Then open `http://localhost:8000/health`.

### Database Foundation

The backend is now prepared for async Postgres usage with:

- `DATABASE_URL` and `DATABASE_ECHO` in settings
- a shared SQLAlchemy async engine created during lifespan startup
- a shared async session factory stored on app state
- a `get_db_session()` dependency for future routes and services

At this step, we have only added the connection layer. ORM models, migrations, and user tables come next.

### Docker Database

The project now includes a standalone Postgres service in the root [docker-compose.yaml](/abs/path/c:/Users/Kayan/OneDrive/Desktop/SE%20Factory/smart_travel_assistant/docker-compose.yaml).

- service name: `db`
- image: `pgvector/pgvector:0.8.2-pg17`
- named volume: `postgres_data`
- init script: `db/init/01-enable-pgvector.sql`

Run only the database with:

```powershell
docker compose up db
```

This keeps Postgres separate from the backend service, which matches the assignment structure. The pgvector extension is created automatically the first time the database volume is initialized.

### First ORM Table

We have now added the first SQLAlchemy ORM model:

- `users` in `app/db/models/user.py`

Originally the backend created tables automatically at startup with `Base.metadata.create_all(...)`. That bootstrap step has been replaced entirely by Alembic migrations - see "Database Migrations (Alembic)" below. `create_all()` is no longer called anywhere; every table, on every DB, is created by a migration.

### Auth Foundation

We have started auth in the smallest useful way:

- `app/schemas/auth.py` defines the request/response data shapes
- `app/core/security.py` handles password hashing and verification

We now also have the first auth route:

- `POST /auth/signup`
- `POST /auth/login`
- `GET /auth/me`

It:

- validates the input with `UserCreate`
- normalizes the email to lowercase
- rejects duplicate emails
- hashes the password before storing it
- returns the created user with `UserRead`
- verifies login credentials against the stored password hash
- rejects invalid credentials with a `401`
- returns a JWT bearer token from login
- resolves the current user from the bearer token on `/auth/me`

The HTTP route stays thin, while the user-creation logic now lives in `app/services/auth.py`.

We are still intentionally keeping auth small. This is now a minimal JWT-based auth flow suitable for the later React frontend.

### Agent Run Persistence

We have now added the first authenticated domain entity after users:

- `agent_runs` in `app/db/models/agent_run.py`

This gives us a simple persisted record of:

- who initiated a run
- what prompt they sent
- what response was stored
- what status the run finished with
- when it happened

There is also a protected route:

- `POST /agent-runs`

It creates a placeholder agent run for the authenticated user and now also creates a linked `tool_log` record. This lets us verify both user-scoped run persistence and tool-level logging before building the full agent/tool workflow.

### Tool Logs

We have now added:

- `tool_logs` in `app/db/models/tool_log.py`

Each tool log belongs to an `agent_run` and stores:

- `tool_name`
- `input_payload`
- `output_payload`
- `status`
- `created_at`

This is the persistence hook we will later use for the classifier, RAG, and live-data tools.

## Database Migrations (Alembic)

Every table in this project - `users`, `agent_runs`, `tool_logs`, `destination_documents`,
`destinations`, `recommendations`, `feedback`, `tag_definitions` - is now created by an Alembic
migration. `Base.metadata.create_all()` has been removed from the app startup path
(`app/core/lifespan.py`) entirely; there is no fallback table creation left.

Migration chain, oldest first:

1. `0e2bdbc1cc5a_create_destinations_table.py` - `destinations` (+ `CREATE EXTENSION vector`)
2. `5f2b7a3d9c14_create_baseline_tables.py` - `users`, `agent_runs`, `tool_logs`,
   `destination_documents` (the tables that used to rely on `create_all()`)
3. `9e4d1f6a8b02_create_ml_feedback_tables.py` - `recommendations`, `feedback`,
   `tag_definitions`, plus `deleted_at` columns on `destinations` and `agent_runs`

`alembic/env.py` sources `sqlalchemy.url` from `app.core.config.get_settings().database_url` (never
hardcoded) and builds `target_metadata` from both declarative bases in the project
(`app.db.base.Base` and `app.db.models.destination.DestinationCorpusBase`), so `alembic check` and
future autogenerate runs have the full schema to diff against. Migrations so far are hand-written
rather than autogenerated, mainly because pgvector's HNSW index type and vector ops classes aren't
reflected by Alembic's autogenerate.

### Common commands

```powershell
# Apply all pending migrations
uv run alembic upgrade head

# Roll back one migration
uv run alembic downgrade -1

# Create a new migration (edit the generated file - autogenerate won't catch
# pgvector-specific DDL like HNSW indexes)
uv run alembic revision -m "add some_table"

# Show current revision / full history
uv run alembic current
uv run alembic history
```

### Bootstrapping an existing, already-populated DB

If your DB already has `users`, `agent_runs`, `tool_logs`, and `destination_documents` from the old
`create_all()` path (true for every DB that ever ran this app before this change), do **not** run
`alembic upgrade head` directly - migration `5f2b7a3d9c14` would try to `CREATE TABLE users` etc.
and fail with "relation already exists". Instead:

```powershell
# 1. Mark the baseline migration as already applied (it matches your
#    existing create_all()-created schema) WITHOUT running its SQL.
#    Use the explicit revision id, not `head` - `head` is one migration
#    further along (9e4d1f6a8b02) and does need to actually run.
uv run alembic stamp 5f2b7a3d9c14

# 2. Now run the real upgrade: this creates recommendations/feedback/
#    tag_definitions and adds the two new deleted_at columns.
uv run alembic upgrade head
```

If your DB has never run this app before (a true fresh DB), just run `uv run alembic upgrade head`
- it will run all three migrations in order.

## Destination Corpus Ingestion (`destinations` table)

A second, richer destination corpus lives alongside the original `travel_destinations_labeled.csv` /
`destination_documents` RAG table (both left untouched). It now **is** wired into the agent - see
"Destination Recommendation (Pre-filter + Cosine Re-rank)" below for how the trip-planner graph
queries it.

### Schema

`destinations` (Alembic-managed - see "Database Migrations (Alembic)" below):

- `id` (UUID PK), `name`, `country`, `region`, `budget_level` (`low`/`medium`/`high`)
- `details` - the composed text that actually gets embedded
- `raw_sources` (JSONB) - unembedded raw per-source text/data, for future re-composition
- `source_provenance` (JSONB) - which source (or failure) produced each field
- `embedding` (`vector(1024)`), `embedding_model`, `embedding_version`
- `content_hash` - sha256 of `details`, used to skip re-embedding unchanged rows
- Indexes: HNSW on `embedding` (`vector_cosine_ops`), btree on `region` and `budget_level`,
  unique on `(name, country)` (the idempotent upsert key)

### Why `Destination` stays on its own declarative base

`destinations` was the first table in the project managed by Alembic rather than
`Base.metadata.create_all()`, back when `create_all()` still handled everything else. The
`Destination` model (`app/db/models/destination.py`) intentionally lives on its **own**
`DeclarativeBase` (`DestinationCorpusBase`), separate from `app.db.base.Base`. That separation
predates and is independent of `create_all()` having since been removed entirely (see "Database
Migrations (Alembic)" below) - it stays because the two model bases represent two conceptually
separate corpora (the legacy `destination_documents` RAG table vs. the richer `destinations`
corpus), not because of a migration-ownership concern anymore. `alembic/env.py` still builds
`target_metadata` from both bases, so autogenerate sees the full schema either way.

### Sources

- **Wikivoyage** - primary destination prose, scraped with the same extraction logic as the
  existing RAG ingestion (`app/services/rag_ingestion.py::_extract_main_text`, reused directly).
- **OpenTripMap** - POI `kinds` aggregated within a radius of the geocoded destination, appended
  to `details` as a one-line summary; raw kind counts kept in `raw_sources`. Requires a free key
  from [opentripmap.io](https://opentripmap.io/product) in `OPENTRIPMAP_API_KEY`. **Not configured
  in this environment** - the step is skipped (not failed) when the key is blank, and `details`
  composes from Wikivoyage + region line only.
- **Numbeo** - has no free API. Instead of scraping per-city pages (which don't expose a numeric
  index without JS), the pipeline fetches Numbeo's public `rankings_current.jsp` table **once per
  run** (~550 cities) and buckets `budget_level` by quartile of that run's index values
  (`< Q1` = low, `Q1-Q3` = medium, `> Q3` = high). Cities not in that ranking table get
  `budget_level = null` - this is expected, not a bug, for smaller/less-common destinations.
- **Open-Meteo geocoding** - substituted for the spec's "optional GeoNames": it's free, keyless,
  and already used by `services/live_conditions.py`, so it resolves canonical lat/lon (needed for
  the OpenTripMap radius query) without introducing a second geocoding provider.

Every source fetch is retried with exponential backoff (`DESTINATION_MAX_RETRIES`,
`DESTINATION_RETRY_BACKOFF_SECONDS`) and failures are isolated per source per destination - a failed
Wikivoyage fetch does not block OpenTripMap/Numbeo for that destination, and a failed destination
does not block the rest of the run. Embedding failures (e.g. an invalid Voyage key) degrade the
same way: rows are still upserted with `embedding = null` and get picked up automatically on the
next successful run via the content-hash cache.

### Seed manifest

`data/destination_seed_manifest.json` - 219 hand-curated, real destinations (`name`, `country`,
`region`, `wikivoyage_url`), versioned and committed, mirroring `rag_source_manifest.json`'s
pattern. Deliberately does **not** bake in coordinates or OpenTripMap/Numbeo identifiers - those
are resolved dynamically per run so a bad guess never gets committed to the manifest.

### Running ingestion from empty

```powershell
# 1. Bring up Postgres (pgvector image) if it isn't already running
docker compose up -d db

# 2. Apply migrations (creates every table, including destinations - see
#    "Database Migrations (Alembic)" above for the existing-DB bootstrap
#    sequence if this isn't a fresh database)
uv run alembic upgrade head

# 3. Set VOYAGE_API_KEY (required) and optionally OPENTRIPMAP_API_KEY in .env

# 4. Smoke-test on a handful of destinations first
uv run python scripts/ingest_destinations.py --limit 5

# 5. Full run (219 destinations; respects VOYAGE_REQUESTS_PER_MINUTE, so budget several minutes)
uv run python scripts/ingest_destinations.py
```

Re-running is always safe: the upsert key is `(name, country)`, and unchanged `details` skip
re-embedding entirely via `content_hash`.

## Destination Recommendation (Pre-filter + Cosine Re-rank)

Replaces the earlier SVC travel-style classifier + CSV hand-weighted scorer. Implemented in
`app/services/destination_recommendations.py`, called by the `destination_recommender` tool
(`app/agent/tools/recommendations_tool.py`) from a single graph node
(`recommend_destinations_node` in `app/agent/graph.py`).

1. Embed the raw trip-request prompt with Voyage (`input_type=query`) - not a synthesized
   structured-field sentence, since destination embeddings were built from Wikivoyage prose in the
   same embedding space.
2. Structured SQL pre-filter over `destinations`: budget ceiling (`budget_level <= requested`, OR
   `NULL` - about 35% of the corpus has no Numbeo coverage and would otherwise be starved out),
   region (skipped when the extraction prompt's `"Flexible"` sentinel is used), and a dormant
   required-tags-above-threshold filter (JSONB weight lookup) - inert until clustering Phase 2/3
   supplies real tag names into `tag_definitions`/`destinations.tags`.
3. Cosine re-rank via `Destination.embedding.cosine_distance(...)` (`<=>`), ordered and limited in
   the same SQL statement so `ix_destinations_embedding_hnsw` (`vector_cosine_ops`) is eligible to
   be used.
4. If the filtered query returns fewer than `min_candidates` rows, re-run once with every hard
   constraint dropped (pure cosine rank over the whole corpus) rather than returning an empty
   slate. The response's `used_relaxed_constraints` flag reports whether this happened.

Each result carries a feature snapshot (`cosine_sim`, `tag_match_count`, `budget_delta`,
`region_match`) alongside `score`/`rank_position` - shaped for the (still unwired) `recommendations`
table for a future learning-to-rank feedback loop, not written there yet.

`EXPLAIN ANALYZE` against the real 219-destination corpus:

```
Limit  (cost=112.47..112.50 rows=10 width=24) (actual time=1.370..1.371 rows=10 loops=1)
  ->  Sort  (cost=112.47..113.02 rows=219 width=24) (actual time=1.369..1.370 rows=10 loops=1)
        Sort Key: ((embedding <=> '[...]'::vector))
        Sort Method: top-N heapsort  Memory: 26kB
        ->  Seq Scan on destinations  (cost=0.00..107.74 rows=219 width=24) (actual time=0.020..1.331 rows=219 loops=1)
              Filter: ((deleted_at IS NULL) AND (embedding IS NOT NULL))
Planning Time: 0.191 ms
Execution Time: 1.404 ms
```

At only 219 rows, the Postgres planner chose a sequential scan on `destinations` followed by an
in-memory top-N heapsort, rather than using the HNSW index (`ix_destinations_embedding_hnsw`). This
is the genuinely correct choice for a table this small: a full sequential scan (~112 cost units,
~1.4ms total) is faster than the overhead of seeking through an index structure. The HNSW index
becomes more valuable as the corpus grows; at this size, a seq scan + sort is optimal.

### Data-quality report

Every run writes `artifacts/destinations/data_quality_report.{json,csv}`: destination count per
region, missing-field rates (`budget_level`, `poi_summary`, `wikivoyage_summary`, `embedding`),
`details` length distribution, and a per-source failure count. The committed artifact reflects a
real 5-destination run (Paris, Lyon, Nice, Marseille, Bordeaux) with all three sources and the
embedding provider live - all missing-field rates are 0.0 and `sources_failed_counts` is empty. A
re-run over the same 5 immediately afterwards produced `embedded_count: 0`,
`skipped_embedding_count: 5` - the content-hash cache, confirmed working with real (not synthetic)
embeddings.

Note: an earlier version of the OpenTripMap integration silently mis-parsed the API's default
GeoJSON response (its published schema nests `kinds` under a doubled `properties.properties` key);
`_fetch_opentripmap_pois` now requests `format=json` and reads the documented flat `SimpleFeature`
list instead, verified against the live API.

## ML Feedback Schema (`recommendations`, `feedback`, `tag_definitions`)

Groundwork for learning-to-rank over recommended destinations. `recommendations` and `feedback`
were wired up in the 2026-07-06 session - see "Feedback Data Model (now wired up)" below.
`tag_definitions` remains schema-only, pending clustering Phase 2/3 (see "Destination Clustering"
below).

- **`recommendations`** (`app/db/models/recommendation.py`) - the **full ranked slate** shown for
  an agent run, one row per destination position, not just the destination the user picked. This
  is deliberate: learning-to-rank needs the whole slate (including what was shown but not chosen),
  not just positive examples.
  - `agent_run_id` (FK -> `agent_runs.id`), `destination_id` (FK -> `destinations.id`, a **UUID**,
    matching that table's PK - not an int), `rank_position`, `score`
  - `features` (JSONB, **not null**) - a snapshot of the ranker's feature row *at recommend time*.
    This is the most important column here: weather, prices, and other live signals drift after
    the fact, so if `features` isn't captured at the moment of recommendation, training data
    quietly desyncs from what the model actually saw. Never derive this column lazily from live
    state later.
  - `deleted_at` (soft delete)
- **`feedback`** (`app/db/models/feedback.py`) - a verdict on one `recommendation` row.
  - `recommendation_id` (FK -> `recommendations.id`), `session_uuid` (an anonymous client UUID -
    **not** a `users` FK, so feedback works without an authenticated session), `verdict`
    (`smallint`, `+1`/`-1`, not null)
  - Partial index `ix_feedback_recommendation_id_verdict_not_null` on `(recommendation_id) WHERE
    verdict IS NOT NULL` - currently equivalent to a plain index since `verdict` is `NOT NULL` at
    the column level, kept as specified for forward-compatibility if that constraint is ever
    relaxed (e.g. a withdrawn-feedback state)
  - `deleted_at` (soft delete)
- **`tag_definitions`** (`app/db/models/tag_definition.py`) - human/LLM-readable labels for
  clusters produced by the offline clustering step below.
  - `cluster_id` (unique), `tag_name`, `description` (LLM-generated rationale), `quality_metrics`
    (JSONB - e.g. silhouette score, cluster size, noise ratio)
  - No `deleted_at` - not part of the per-run audit trail the other two tables are.

`recommendations.destination_id` has no ORM `relationship()` to `Destination`: that model lives on
its own declarative base/registry (see "Why `Destination` stays on its own declarative base"
above), so `relationship()` can't resolve it by class name across bases. The FK column and its DB
constraint exist regardless, but this cross-base split has a sharper consequence than just a
missing convenience accessor - see "Inserting `Recommendation` rows: use Core `insert()`, not
`session.add()`" below, discovered when this table's writes were actually wired up.

## Feedback Data Model (now wired up)

As of the 2026-07-06 session, `recommendations` and `feedback` are no longer schema-only:

- **`recommendations`**: one row per candidate in the ranked slate the pre-filter+cosine node
  returned for a given `agent_runs` row - not just the top result. `features` is a JSONB snapshot
  captured once, at recommend time (`cosine_sim`, `tag_match_count`, `budget_delta`,
  `region_match`) - it is never recomputed later, so it stays an honest record of what the model
  saw at decision time even if the underlying destination's data changes afterward. Persisted in
  `app/services/agent_runs.py::create_agent_run`, right after the `agent_runs` row is committed
  (its `id` is the FK target, so persistence can only happen after that commit) - see
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
  useful for future negative sampling or exposure-weighting, not yet consumed by anything. Nothing
  currently trains on this table; populating it is the whole point of this session's work,
  actually wiring up a learning-to-rank model on top of it is still future work.

### Inserting `Recommendation` rows: use Core `insert()`, not `session.add()`

Discovered while wiring up persistence (not previously exercised - nothing had ever written a
`Recommendation` row via the ORM before this session): `Recommendation.destination_id`'s
`ForeignKey("destinations.id")` can **never** resolve through SQLAlchemy's ORM metadata lookup,
because `Destination` lives on a separate `DeclarativeBase`/`MetaData` (`DestinationCorpusBase`)
from `Recommendation` (`app.db.base.Base`). A normal ORM write (`session.add()`/`session.add_all()`
followed by `session.commit()`, or an ORM bulk-insert given a list of param dicts) triggers
SQLAlchemy's flush-time `_sorted_tables` dependency-graph computation, which needs to resolve
**every** FK target of the flushed mapper's table - including ones outside the current flush - and
raises `sqlalchemy.exc.NoReferencedTableError` when it can't.

The fix, used in `persist_recommendation_slate`: bake every row into a single Core
`insert(Recommendation).values([...]).returning(Recommendation)` statement (not `session.add()`,
and not passing rows as a separate `execute()` params list - that still hits the same bulk-insert
code path) - this bypasses the ORM flush machinery entirely while still returning fully-populated
`Recommendation` instances via `RETURNING`. Read-only queries (a plain `select()` with an explicit
`.join()` onclause, like `get_recommendations_for_agent_run`) are unaffected - this is a
flush/write-path-only issue. Do not "fix" this by changing `Recommendation`'s FK declaration or
merging the two declarative bases - that's a real, deliberate conceptual split (see "Why
`Destination` stays on its own declarative base" above), not a mistake to undo.

## Learning-to-Rank: Cold-Start Bootstrap Ranker

`recommend_destinations()` (`app/services/destination_recommendations.py`) retrieves candidates
via a structured SQL pre-filter + pgvector cosine re-rank, exactly as before. Optionally, when
`RANKER_ENABLED=true` **and** a trained model exists at `artifacts/ranker/model.joblib`, the
cosine-retrieved candidate set is re-ordered by a LightGBM `LGBMRanker` before truncation to
`limit` - cosine stays the retrieval mechanism; the ranker only reorders. `score` in the API
response always means cosine similarity, regardless of which order the ranker produced; only
`rank_position` reflects the ranker's order. Because reordering happens **before** truncation, a
candidate the ranker considers more relevant can be promoted into the final slate even if it
wasn't in the cosine-only top-`limit` - e.g. with `limit=5`, cosine order might put destinations
`[A, B, C, D, E, ...]` but the ranker's top 5 could be `[A, B, C, F, G]`, `D`/`E` bumped out in
favor of `F`/`G` from further down the retrieved set. This was confirmed during implementation:
with no truncation the reordered set is identical to cosine's, only sequence differs; with
truncation, the final slate's membership can (and did) differ.

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
practice (confirmed by the bootstrap-trained model's own feature importances: `cosine_sim` ~13x
higher than `tag_match_count`, the next-highest). `RANKER_ENABLED` therefore defaults to `false`.

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
On the checked-in bootstrap dataset (150 synthetic queries, 2250 candidate rows): `NDCG@3=0.916`,
`NDCG@5=0.929`, feature importances `cosine_sim=5306, tag_match_count=406, budget_delta=244,
region_match=44`.

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

Every model in `Recommendation`/`AgentRun`'s relationship graph (`AgentRun`, `User`, `ToolLog`, ...)
must be imported somewhere before the first query touches either, or SQLAlchemy raises
`InvalidRequestError: ... failed to locate a name (...)` trying to resolve a `relationship()`
string lazily - `scripts/train_ranker.py` imports every model module for this side effect, the
same pattern `alembic/env.py` already uses for autogenerate.

### Config

`RANKER_ENABLED` (`backend/.env`, default `false`) - `recommend_destinations()` only re-ranks when
this is `true` **and** `artifacts/ranker/model.joblib` exists; otherwise cosine order is used,
identical to before this feature existed.

## Destination Clustering (`scripts/cluster_destinations.py`)

Offline HDBSCAN soft clustering over `destinations.embedding`, producing weighted travel-style
tags per destination (`destinations.tags`) and human-readable cluster labels (`tag_definitions`).
Not wired into the agent or any route - a standalone, run-once(-per-corpus-change) script, never a
graph node. Requires the corpus to already be ingested with embeddings (see "Destination Corpus
Ingestion" above) - **aborts if fewer than 50 destinations have a non-null embedding**, since
HDBSCAN is meaningless on a tiny corpus.

### Why this approach

- **Cosine geometry, L2-normalized.** Retrieval elsewhere in this project (RAG, destination
  similarity) uses cosine distance, so embeddings are L2-normalized before UMAP and UMAP is fit
  with `metric="cosine"`, keeping the clustering geometry consistent with how these vectors are
  used everywhere else.
- **UMAP before HDBSCAN, not HDBSCAN directly on 1024-dim embeddings.** HDBSCAN's density
  estimates degrade in high dimensions (the curse of dimensionality flattens pairwise distances).
  UMAP reduces to ~10 dimensions first (`--umap-n-components`, default 10) on a fixed
  `random_state`, and HDBSCAN then runs in Euclidean space on that reduced embedding
  (`metric="euclidean"`, `cluster_selection_method="eom"`).
  - `HDBSCAN` has **no `random_state` of its own** - its only source of run-to-run randomness is
    the approximate minimum-spanning-tree algorithm, which is disabled here
    (`approx_min_span_tree=False`) so the main clustering run is fully reproducible for a fixed
    UMAP embedding. This also means the stability check below is measuring exactly one thing:
    sensitivity to UMAP's random initialization.
- **Soft clustering, not hard labels.** `hdbscan.all_points_membership_vectors` gives every
  destination (including HDBSCAN's hard-label noise points) a membership weight against every
  cluster. Weights above `--membership-threshold` (default `0.15`) become a destination's weighted
  `{cluster_id: weight}` tags - a destination can legitimately carry multiple travel-style tags.
- **Two independent quality signals, not one.** Silhouette score (computed in UMAP space,
  excluding noise - undefined with fewer than 2 clusters) measures separation; DBCV
  (`hdbscan.validity.validity_index`, wrapped in a fallback since it can fail on degenerate inputs)
  is density-aware and purpose-built for HDBSCAN's variable-density clusters. Both land in
  `quality_report.json` rather than picking one.
- **Stability is measured, not assumed.** `cluster` re-fits the entire UMAP -> HDBSCAN pipeline
  across `--n-stability-runs` (default 5) different UMAP seeds and reports the pairwise Adjusted
  Rand Index between the resulting label sets. A mean ARI below `0.7` is flagged
  `"flagged_unstable": true` in `stability_report.json` - a signal to retune `--min-cluster-size`
  or gather more corpus data, not a hard failure.

### Three phases, separately resumable

```powershell
# Phase 1: fit UMAP + HDBSCAN, write weighted {cluster_id: weight} tags to
# destinations.tags, write all artifacts/clustering/ outputs.
uv run python scripts/cluster_destinations.py cluster

# Phase 2: ask Claude (ANTHROPIC_STRONG_MODEL) to propose a tag_name +
# description per cluster from artifacts/clustering/ (no re-clustering).
# Upserts into tag_definitions - re-run any time to regenerate proposals.
uv run python scripts/cluster_destinations.py name

# Review/edit tag_definitions.tag_name / .description in the DB, then:

# Phase 3: rewrite destinations.tags from cluster_id keys to the approved
# tag_name keys. Re-runnable any time tag_definitions changes.
uv run python scripts/cluster_destinations.py apply-tags
```

`cluster` accepts `--min-cluster-size` (default 7), `--min-samples` (defaults to
`--min-cluster-size` when omitted, HDBSCAN's own default), `--membership-threshold` (default
`0.15`), `--umap-n-components`/`--umap-n-neighbors`/`--umap-min-dist`, `--random-state` (default
`42`), `--n-stability-runs` (default 5, `--skip-stability` to skip), and `--dry-run` (compute and
write artifacts without touching `destinations.tags` - useful while tuning hyperparameters).

The source of truth for `name` and `apply-tags` is `artifacts/clustering/cluster_members.json`
(written by `cluster`), not the live `destinations.tags` column - `apply-tags` is safe to re-run at
any point in the naming/approval process, and clusters without an approved `tag_definitions` entry
are simply omitted from the written tags rather than blocking the run.

### Reading `artifacts/clustering/`

- `quality_report.json` - `n_clusters`, `noise_ratio`, `hard_cluster_sizes` (HDBSCAN's hard label
  counts) vs. `soft_cluster_sizes` (count above `membership_threshold` per cluster - these can
  differ, which is the point of soft clustering), `silhouette_umap_space`, `dbcv`.
- `stability_report.json` - `mean_ari`/`min_ari` across `n_runs` UMAP seeds, `flagged_unstable`.
  Low ARI means small changes to UMAP's initialization meaningfully change which destinations end
  up in which cluster - treat the clustering as provisional until this improves.
- `cluster_members.json` - every destination's raw membership weight for every cluster, sorted
  descending, for manual inspection. This is what `name` and `apply-tags` actually read.
- `membership_vectors.npz` - the raw `all_points_membership_vectors` output (`destination_ids`,
  `membership` matrix, `labels`), for anyone who wants to re-analyze without re-fitting.
- `umap_reducer.joblib` / `hdbscan_clusterer.joblib` - the fitted objects, for reproducibility.
- `umap_scatter.png` - a **dedicated 2D UMAP fit** (not a 2D slice of the clustered
  `n_components`-dim embedding) colored by final cluster assignment.
- `membership_weight_histogram.png` - distribution of nonzero soft-membership weights, a sanity
  check on whether `--membership-threshold` is in a reasonable place.
- `naming_prompts/cluster_<id>.json` - the exact example destinations, quality metrics, and Claude
  proposal for every cluster's naming call, for reproducibility.

### When to re-cluster

Re-run the full `cluster` -> `name` -> `apply-tags` sequence whenever the destinations corpus
changes meaningfully (a full re-ingestion, a large batch of new destinations, or a change to the
embedding model/version). `cluster` is idempotent - it overwrites `destinations.tags` and every
artifact cleanly on each run - but it is not incremental: adding a handful of destinations to an
already-clustered corpus means re-fitting from scratch, not assigning the new rows to existing
clusters.

## Provider-Agnostic LLM Layer

All three LLM call sites (field extraction, trip synthesis, offline cluster naming) go through
`app/services/llm_providers/`'s `LLMProvider` protocol rather than calling either vendor directly.
The interface is a single method:

```python
async def complete(self, messages: list[Message], **opts: object) -> str: ...
```

`messages` is a list of `{"role": "system" | "user", "content": str}` dicts. `**opts` lets a call
site override `max_tokens`/`temperature` for that one call (`extract_request_fields` and
`propose_cluster_tag` both do this; `synthesize_trip_response` doesn't, and falls back to the
provider's configured defaults). `app/services/llm.py`'s three orchestration functions
(`extract_request_fields`, `synthesize_trip_response`, `propose_cluster_tag`) only depend on this
protocol, never on a specific provider's SDK or wire format.

**No fast/strong model tiers** (removed 2026-07-06 - `ModelTier`, `choose_model()`,
`fast_model_name()`/`strong_model_name()`/`resolve_model_name()` all deleted). Each provider always
uses one single configured model (`gemini_model` / `anthropic_model` in `Settings`) for every call
site. This was a deliberate simplification, not an oversight: the two-tier routing existed to send
cheap/high-volume calls (extraction) to a fast model and quality-sensitive calls (synthesis,
naming) to a stronger one, but the "strong" side of that split (`gemini-3.1-pro`) turned out to be
a broken model string - it doesn't exist as a callable model at all (confirmed via
`client.models.list()` - only `gemini-3.1-pro-preview` does) - and separately, the real
Gemini-branded models require a paid prepay balance once billing is enabled on the project, while
Gemma models are served for free through the same API. Rather than juggling two tiers across two
different billing/availability situations, `gemini_model` is pinned to a single free Gemma 4 model
for now. See "Gemini (default provider)" below for the full live-testing story.

### Why an abstraction at all (lock-in, cost)

The **provider abstraction** itself (not the tiering, which is gone) is still motivated by a real
incident from an earlier session: the Anthropic API key on this project ran out of credit
mid-session (see `.claude/memory/` for the incident) with no fallback - every LLM call in the app
failed at once. A second, independently-billed provider behind the same interface means a billing
problem with one vendor doesn't take down request field extraction and trip synthesis together.

`LLM_PROVIDER` (`anthropic` or `gemini`, **default `gemini`**) is one global switch - it is not a
per-call-site setting and there is no automatic fallback between providers. Set both providers'
credentials if you want to be able to switch without restarting with different env vars, or just
the one you're using.

### How to switch providers

Set one env var and restart - no code changes:

```
LLM_PROVIDER=gemini     # or: anthropic
```

Everything else (`GEMINI_MODEL` vs `ANTHROPIC_MODEL`, per-provider `_MAX_TOKENS`/`_TEMPERATURE`) is
already configured for both providers side by side in `.env`/`.env.example`, so the inactive
provider's settings just sit unused rather than needing to be added when you switch.

### Gemini (default provider) - currently pinned to Gemma 4, live-verified working

Implemented in `app/services/llm_providers/gemini_provider.py` using the official
[`google-genai`](https://pypi.org/project/google-genai/) Python SDK (`client.aio.models.generate_content`)
rather than raw REST. Configure:

```
LLM_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemma-4-26b-a4b-it   # free tier - see below for why this, not a gemini-3.1-* model
```

Generate a **restricted** API key (scoped to the Generative Language API) at
[Google AI Studio](https://aistudio.google.com/apikey) - Google is phasing out unrestricted
Gemini API keys during 2026 (restricted keys work until September 2026; after that, only
service-account-bound auth keys are accepted).

**Live-testing story (2026-07-06), in the order it actually happened - useful if you hit any of
these again:**
1. `403 PERMISSION_DENIED (SERVICE_DISABLED)` - the key's Google Cloud project never had the
   Generative Language API enabled. Fixed by enabling it in Cloud Console (a project-level
   setting, not a key-level one).
2. `403 PERMISSION_DENIED (API_KEY_SERVICE_BLOCKED)` - a *different* 403, after the project-level
   API was enabled: the key itself had an API restriction not including the Generative Language
   API. Fixed on the key's own "API restrictions" list in Cloud Console.
3. `429 RESOURCE_EXHAUSTED` ("Your prepayment credits are depleted") on `gemini-3.1-flash-lite` -
   this is a *third*, separate system from the rate-limit quota counters (RPM/TPM/RPD) shown on
   the AI Studio usage page: a prepay dollar balance that gets debited per token on every call to a
   Gemini-branded model, independent of whether you're anywhere near your quota ceiling. Adding
   prepay credits fixed it - confirmed live success on `gemini-3.1-flash-lite`,
   `gemini-3.1-pro-preview`, `gemini-2.5-flash`, `gemini-2.5-pro`, and others once credits landed.
4. **Gemma models never hit any of the above** - confirmed live success on `gemma-4-26b-a4b-it`/
   `gemma-4-31b-it` *while* the Gemini-branded models were still blocked on step 3's billing wall.
   That's the evidence `gemma-4-26b-a4b-it` is genuinely free, not just cheap.
5. `gemini_strong_model`'s old default, `gemini-3.1-pro`, was independently confirmed to be a
   **nonexistent model string** (`404 NOT_FOUND`) via `client.models.list()` - the real model is
   `gemini-3.1-pro-preview`. Moot now that tiers are gone, but worth knowing if `GEMINI_MODEL` is
   ever pointed back at a `gemini-3.1-*` model by name.

So: live Gemini API calls **are** confirmed working now (both Gemma and Gemini-branded models) -
this supersedes an earlier version of this doc that said no live call had been verified yet.
`GEMINI_MODEL` stays pinned to the free Gemma model so running this app doesn't silently spend
money; switching to a `gemini-3.1-*` model is a one-line config change once billing is set up the
way you want it.

**Transport tradeoff, deliberately accepted:** every other HTTP call in this app reuses one shared
`httpx.AsyncClient` from `app/core/lifespan.py` (see `CLAUDE.md`'s async conventions). The
`google-genai` SDK manages its own transport internally and cannot be handed that shared client.
`GeminiProvider` works around this by memoizing the SDK's `genai.Client` per API key
(`functools.lru_cache` in `gemini_provider.py`) so it's constructed once per process rather than
once per call, not once per request - but it is still a second, separate connection pool from the
rest of the app's outbound HTTP traffic.

### Anthropic (kept in place, currently unused)

Implemented in `app/services/llm_providers/anthropic_provider.py` via plain REST over the shared
`httpx.AsyncClient` - no Anthropic SDK dependency. Selecting it back is a config change:

```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-haiku-4-5
```

### Adding a third provider

Implement the `LLMProvider` protocol (`app/services/llm_providers/protocol.py`) as a new module in
`app/services/llm_providers/` - one `complete()` method translating the provider-agnostic
`messages` call into that provider's own request shape - and add it to `get_llm_provider()`'s
dispatch (`app/services/llm_providers/factory.py`). `app/services/llm.py`'s three orchestration
functions need no changes.
