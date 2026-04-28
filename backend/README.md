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

For now, the backend creates tables automatically at startup with `Base.metadata.create_all(...)`. This is a good early-project bootstrap step so the schema appears immediately in pgAdmin once the backend connects to the running database. Later, when the schema grows, we should replace this with Alembic migrations.

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
