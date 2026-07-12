# Multi-turn clarification loop in the trip-planning graph

Date: 2026-07-12

## Problem

The trip-planning LangGraph pipeline (`app/agent/graph.py`) extracts structured fields
(`destination_name`, `location_query`, `location_country_code`, `travel_profile`) from the user's
prompt once, via a single LLM call, then proceeds regardless of how incomplete the result is.
Missing fields are either silently guessed by the LLM or left null. There is no mechanism for the
agent to ask the user a follow-up question and incorporate the answer before recommending a
destination.

Goal: extraction becomes a loop that pauses the graph and asks the user directly when the
information needed for a *good* recommendation is missing, resumes with their answer, and repeats
(bounded) until satisfied.

## Current state (as of this design)

- **Graph compilation**: `build_trip_planner_graph()` (`graph.py:489-507`) calls `graph.compile()`
  with **no checkpointer**. `run_trip_planner()` (`planner.py:32-44`) calls `graph.ainvoke(...)`
  once, synchronously, with no `config`/`thread_id`. No checkpointer of any kind exists anywhere in
  the codebase today.
- **Non-serializable runtime in state**: `TripPlannerState` (`graph.py:16-29`) carries
  `tool_registry` and `tool_context` directly — `tool_context` wraps a shared `httpx.AsyncClient`
  and a per-request SQLAlchemy `AsyncSession`. This only works today because there's no
  checkpointer trying to serialize it. Adding one requires splitting checkpointed state from
  per-invocation runtime.
- **Extraction schema** (`schemas/claude.py`): `ExtractedRequestFields` has `destination_name`,
  `location_query`, `location_country_code` (all optional) and `travel_profile: TravelProfile |
  None`. `TravelProfile` is all-or-nothing — every field (`region`, `budget_level`,
  `tourism_level`, `has_hiking`, `has_beach`, `culture_score`, `luxury_score`, `family_friendly`,
  `nightlife_level`, `avg_temp_peak`) is required once the object is present. There is no
  date/trip-length/traveler-count field, and nothing downstream would consume one (`live_conditions`
  is current weather, not a forecast) — not adding one.
- **Recommender is tolerant, not blocking**: `recommend_destinations_node` (`graph.py:254-259`)
  already runs on `query_text=prompt` alone; `budget_level`/`region` are optional SQL pre-filters.
  So "required to proceed" is a quality bar, not a crash-prevention one.
- **Tag-based filtering is currently inert for this purpose**: `TravelProfile`'s activity/style
  fields (`has_hiking`, `has_beach`, `culture_score`, `luxury_score`, `family_friendly`,
  `nightlife_level`) are extracted but never forwarded into
  `DestinationRecommendationRequest.required_tags` (only `budget_level`/`region` are forwarded
  today). Separately, `tag_definitions` currently holds exactly 5 rows from a real, completed
  offline clustering run: "South American Cultural Heritage", "European Architectural Heritage",
  "Asian Cultural Heritage", "Oceania Cultural Heritage", "Dynamic Urban Metropolises" — broad
  regional/cultural clusters from unsupervised HDBSCAN clustering, with no relationship to
  hiking/beach/nightlife/family axes. Mapping `has_hiking=true` onto a `required_tags` filter would
  be a no-op at best: `destination_recommendations.py:133-136` filters
  `WHERE tags[tag] >= threshold`, and no destination has a `"hiking"` key, so every row gets
  excluded and the `min_candidates` fallback silently drops all filters. A real activity-tag
  taxonomy would require redesigning/re-running the offline clustering pipeline — out of scope
  here.

## Design

### 1. State & graph structure

Split `TripPlannerState` into:
- **Checkpointed (serializable)**: `prompt`, extracted fields, `clarification_turn: int`,
  `clarification_qa: list[{question, answer}]`, `status`, `response_sections`,
  `recommended_destinations`, `final_response`, `tool_logs`.
- **Per-invocation runtime**: `tool_registry`, `tool_context` (and transitively `http_client`,
  `session`) — passed via LangGraph's `Runtime[Context]` mechanism, supplied fresh on every
  `ainvoke` call (both the initial call and every resume), never checkpointed.

New node `clarify_missing_fields`, inserted between `extract_request_fields` and
`recommend_destinations`:
- If the required bar (below) is satisfied, or `clarification_turn >= 3`, passes straight through
  to `recommend_destinations` (logging a tool_log noting why, if the cap is what let it through).
- Otherwise builds one question (priority rules below), calls
  `interrupt({"question": ..., "turn": n})` to pause the graph, and on resume merges the returned
  answer into `prompt` as additional context text, increments `clarification_turn`, and routes back
  to `extract_request_fields` for another extraction pass over the enriched prompt.

Graph edges: `extract_request_fields -> clarify_missing_fields ->` (conditional: back to
`extract_request_fields`, or forward to `recommend_destinations`).

Compile with `InMemorySaver` as a lifespan singleton (`app.state.resources["checkpointer"]`),
passed to `graph.compile(checkpointer=...)`. Each fresh run gets `thread_id = uuid4()`; resumes
reuse it. **Known limitation, accepted for this deliverable**: in-memory means state does not
survive a process restart and is not shared across multiple workers. Acceptable because
docker-compose runs a single backend container and nothing in this repo requires multi-worker
support today. Revisit with a Postgres-backed checkpointer if that changes.

### 2. Required-signal definition and question priority

**Required bar** — both of the following, not just one:
1. A destination anchor: `destination_name` or `location_query`.
2. A usable preference profile: `travel_profile` present, with `region != "Flexible"` and
   `budget_level` set.

**Question priority per round**: only destination/region/budget (SQL filters) and the
activity/style descriptors (via query enrichment, below) actually move the recommendation output —
so each round asks about whichever required-bar category is missing, destination-anchor first if
both are missing (single highest-value question, and it lets the recommender skip the region SQL
filter entirely). If only the profile is missing, the exact question depends on *which* sub-case
that is, since `TravelProfile` is all-or-nothing:
- `travel_profile is None` (LLM couldn't infer anything): ask one combined question covering budget
  and general travel style, e.g. "What's your budget, and what kind of trip are you looking for
  (hiking, beach, culture, nightlife, family, or luxury)?" — `budget_level` and the activity fields
  are all unset, so one question gathering all of it is appropriate.
- `travel_profile` present but `region == "Flexible"` (the LLM's null-sentinel): `budget_level` and
  the activity fields are already filled in (even if only default-ish guesses), so re-asking about
  them would be redundant. Ask specifically about region instead: "Do you have a specific region or
  country in mind, or should I keep the search worldwide?"

Cap: 3 rounds. If still unsatisfied at the cap, proceed best-effort with whatever's been gathered
(logging that the cap was hit) rather than looping forever.

### 3. Query enrichment (makes gathered profile info actually change results)

In `recommend_destinations_node`, before calling the recommender, build an enriched query string
from whichever `travel_profile` fields are set (e.g. `has_hiking` -> "enjoys hiking and outdoor
trails", `culture_score >= 7` -> "seeks rich cultural experiences", similarly for
beach/luxury/family/nightlife) and append these phrases to `state["prompt"]` as the `query_text`
sent for embedding, instead of the raw prompt. `budget_level`/`region` continue through the
existing SQL pre-filter unchanged. `required_tags` stays empty/unused — inert against the current
5-tag corpus, not touched by this work.

### 4. API contract and persistence

Extend `AgentRunCreate` with `thread_id: str | None = None` and
`clarification_answer: str | None = None`, with a model validator: `thread_id` set requires
`clarification_answer` set (resume call); `thread_id` absent means `clarification_answer` must be
absent too (fresh call).

Response is a union: existing `AgentRunRead` for a completed run, or new
`AgentRunNeedsInput {status: "needs_input", thread_id, question, turn}` when the graph pauses.
Route returns 200 for `needs_input`, 201 for a completed run (set via the `Response` param, not the
decorator's fixed `status_code`).

**No DB row until completion.** `thread_id` is a `uuid4()` key into the in-memory checkpointer,
independent of `AgentRun.id`. `run_trip_planner`/`create_agent_run` only run the existing
persistence flow (DB row, tool_logs, recommendations, Discord webhook) once the graph reaches
`END`. This requires no schema/migration change.

**Logging falls out of the existing pattern for free.** `tool_logs` is a checkpointed state field,
so entries from every clarification round (including interrupted ones) accumulate across resumes
automatically. When the run completes, `create_agent_run`'s existing loop over
`planner_result.tool_logs` persists all of them at once, in order — no service-layer changes
needed. Each round logs `tool_name="clarification_loop"`, `output_payload` = the question asked,
`status="needs_input"` while paused, `"completed"` once satisfied, `"skipped"` if it passed through
because the cap was hit.

### 5. Frontend

Extend `PlannerRequest` with optional `thread_id?: string` and `clarification_answer?: string`.
Add `AgentRunNeedsInput` type; `createAgentRun` returns `AgentRunRead | AgentRunNeedsInput`,
narrowed on `status`.

New `App.tsx` state: `clarification: {threadId, question, turn} | null` and
`clarificationAnswer: string`. `handlePlanSubmit` sets `clarification` instead of `result` when the
response is `needs_input`. New `handleClarificationSubmit` re-calls `createAgentRun` with
`{prompt, retrieval_top_k, thread_id, clarification_answer}`, looping into another `clarification`
update if still unsatisfied, or into `setResult` once complete.

UI: while `clarification` is set, render a chat-style block above the results area — the original
prompt, the agent's question as a message bubble, a single text input + "Answer" button, and a
persistent "Still forming your trip plan..." badge (reusing the existing `gt-pill` style). The main
prompt form stays disabled/hidden during this so it reads as a continuation of the same request
thread, not a new submission or a form reload.

### 6. Tests

- Extraction-incomplete -> interrupt -> resume -> complete: mock the LLM extraction call (per
  `conftest.py` convention) to return an incomplete result first; assert the route responds
  `needs_input` with a `thread_id` and question. POST again with that `thread_id` + an answer, mock
  extraction to return a complete result; assert a normal `AgentRunRead` with
  `status="completed"` and a persisted `AgentRun` row (only after this second call, not before).
- Max-turns cap: mock extraction to stay incomplete across all rounds; assert the 4th call (past
  the 3-round cap) proceeds best-effort to a completed run rather than interrupting again, and that
  `tool_logs` contains an entry noting the cap was hit.
- Query enrichment: unit test on `recommend_destinations_node` (or the helper it calls) asserting
  that a `travel_profile` with `has_hiking=True`/`culture_score=9` produces a `query_text`
  containing the expected enrichment phrases, and an empty/absent profile leaves `query_text`
  unchanged.

## Out of scope

- Per-field confidence scoring on extraction (v1 keys on missing required fields, not ambiguity).
- A real activity-tag taxonomy / re-clustering to cover hiking/beach/nightlife/family axes.
- Postgres-backed (multi-worker/restart-durable) checkpointing.
- Persisting a placeholder `AgentRun` row for interrupted runs (would need a schema/migration
  change to make `response` nullable; explicitly declined).
