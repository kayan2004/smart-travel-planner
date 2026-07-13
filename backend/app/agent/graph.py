import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from app.agent.tools.base import ToolContext
from app.agent.tools.registry import ToolRegistry
from app.schemas.claude import TravelProfile
from app.schemas.live_conditions import LiveConditionsRequest
from app.schemas.rag_retrieval import RagRetrievalRequest
from app.schemas.recommendations import DestinationRecommendationRequest
from app.services.llm import extract_request_fields, synthesize_trip_response
from app.services.llm_providers import LLMAuthenticationError

logger = logging.getLogger(__name__)


@dataclass
class TripPlannerRuntime:
    """Non-serializable, per-invocation dependencies - deliberately kept
    out of TripPlannerState. tool_context carries a Settings object that,
    for a BYOK request, has a user-supplied API key on it (see
    app/core/byok.py); that key must never enter checkpointable graph
    state (the graph is compiled with no checkpointer today, but this
    keeps it that way structurally rather than by omission - the moment a
    checkpointer is ever added, tool_context already lives outside what it
    would serialize).
    """

    tool_registry: ToolRegistry | None
    tool_context: ToolContext | None


class TripPlannerState(TypedDict):
    prompt: str
    travel_profile: NotRequired[TravelProfile | None]
    destination_name: NotRequired[str | None]
    location_query: NotRequired[str | None]
    location_country_code: NotRequired[str | None]
    retrieval_top_k: int
    status: str
    response_sections: list[str]
    recommended_destinations: NotRequired[list[dict[str, Any]]]
    final_response: NotRequired[str | None]
    tool_logs: list[dict[str, str]]


def initialize_trip_state(state: TripPlannerState) -> TripPlannerState:
    prompt = state["prompt"].strip()
    return {
        **state,
        "prompt": prompt,
        "status": "completed",
        "response_sections": [f"Prompt: {prompt}"],
        "recommended_destinations": [],
        "final_response": None,
        "tool_logs": [],
    }


async def extract_request_fields_node(
    state: TripPlannerState, runtime: Runtime[TripPlannerRuntime]
) -> TripPlannerState:
    tool_logs = list(state["tool_logs"])
    response_sections = list(state["response_sections"])
    tool_context = runtime.context.tool_context

    if (
        state.get("travel_profile") is not None
        and state.get("destination_name") is not None
        and state.get("location_query") is not None
    ):
        tool_logs.append(
            {
                "tool_name": "request_field_extractor",
                "input_payload": state["prompt"],
                "output_payload": "Extraction skipped because the request already included the main structured fields.",
                "status": "skipped",
            }
        )
        return {"tool_logs": tool_logs}

    if tool_context is None or tool_context.http_client is None:
        tool_logs.append(
            {
                "tool_name": "request_field_extractor",
                "input_payload": state["prompt"],
                "output_payload": "Extraction skipped because the Claude runtime is unavailable.",
                "status": "skipped",
            }
        )
        return {"tool_logs": tool_logs}

    try:
        extracted = await extract_request_fields(
            tool_context.http_client,
            tool_context.settings,
            prompt=state["prompt"],
        )
    except LLMAuthenticationError:
        if tool_context.is_byok:
            # A BYOK key being rejected must surface as a clear error to
            # the caller, not degrade into a "partial" success - re-raise
            # so it propagates up to the route, which converts it to a 4xx.
            raise
        # The server's own default key failing is an ops problem, not a
        # user-facing error - keep degrading gracefully like any other
        # tool failure.
        tool_logs.append(
            {
                "tool_name": "request_field_extractor",
                "input_payload": state["prompt"],
                "output_payload": "Extraction failed: the configured LLM API key was rejected.",
                "status": "failed",
            }
        )
        response_sections.append(
            "Request field extraction failed, so the agent continued with only the explicit request fields."
        )
        return {
            "status": "partial" if state["status"] == "completed" else state["status"],
            "response_sections": response_sections,
            "tool_logs": tool_logs,
        }
    except Exception as exc:
        logger.exception("Request field extraction failed")
        tool_logs.append(
            {
                "tool_name": "request_field_extractor",
                # Exception type only, not str(exc) - the full message
                # (which for a DB/HTTP/provider error can carry internal
                # details) goes to the server log above, not to
                # tool_logs.output_payload, which flows straight into the
                # API response and the frontend's visible "Tool trail".
                "input_payload": state["prompt"],
                "output_payload": f"Extraction failed: {type(exc).__name__}.",
                "status": "failed",
            }
        )
        response_sections.append(
            "Request field extraction failed, so the agent continued with only the explicit request fields."
        )
        return {
            "status": "partial" if state["status"] == "completed" else state["status"],
            "response_sections": response_sections,
            "tool_logs": tool_logs,
        }

    merged_destination_name = state.get("destination_name") or extracted.destination_name
    merged_location_query = state.get("location_query") or extracted.location_query
    merged_country_code = (
        state.get("location_country_code") or extracted.location_country_code
    )
    merged_travel_profile = state.get("travel_profile") or extracted.travel_profile

    inferred_fields: list[str] = []
    if state.get("destination_name") is None and extracted.destination_name is not None:
        inferred_fields.append(f"destination={extracted.destination_name}")
    if state.get("location_query") is None and extracted.location_query is not None:
        inferred_fields.append(f"location_query={extracted.location_query}")
    if (
        state.get("location_country_code") is None
        and extracted.location_country_code is not None
    ):
        inferred_fields.append(f"country_code={extracted.location_country_code}")
    if state.get("travel_profile") is None and extracted.travel_profile is not None:
        inferred_fields.append("travel_profile=inferred")

    if inferred_fields:
        response_sections.append(
            "Inferred request fields from the prompt: " + ", ".join(inferred_fields)
        )

    tool_logs.append(
        {
            "tool_name": "request_field_extractor",
            "input_payload": state["prompt"],
            "output_payload": json.dumps(extracted.model_dump(mode="json")),
            "status": "completed",
        }
    )

    return {
        "destination_name": merged_destination_name,
        "location_query": merged_location_query,
        "location_country_code": merged_country_code,
        "travel_profile": merged_travel_profile,
        "response_sections": response_sections,
        "tool_logs": tool_logs,
    }


async def retrieve_context_node(
    state: TripPlannerState, runtime: Runtime[TripPlannerRuntime]
) -> TripPlannerState:
    tool_logs = list(state["tool_logs"])
    response_sections = list(state["response_sections"])
    tool_registry = runtime.context.tool_registry
    tool_context = runtime.context.tool_context
    status = state["status"]
    retrieval_input = {
        "query": state["prompt"],
        "destination_name": state.get("destination_name"),
        "top_k": state["retrieval_top_k"],
    }

    if tool_registry is None or tool_context is None:
        tool_logs.append(
            {
                "tool_name": "destination_context_retriever",
                "input_payload": state["prompt"],
                "output_payload": (
                    "RAG retrieval could not run because shared services are unavailable."
                ),
                "status": "failed",
            }
        )
        response_sections.append(
            "Some live tool services were unavailable, so this run was only partially completed."
        )
        return {
            "status": "partial",
            "response_sections": response_sections,
            "tool_logs": tool_logs,
        }

    try:
        retrieval = await tool_registry.get("destination_context_retriever").arun(
            RagRetrievalRequest(**retrieval_input),
            tool_context,
        )
        if retrieval.results:
            top_destinations = ", ".join(
                f"{item.destination_name} ({item.similarity_score})"
                for item in retrieval.results[:3]
            )
            response_sections.append(
                f"Relevant destination context: {top_destinations}"
            )
            tool_logs.append(
                {
                    "tool_name": "destination_context_retriever",
                    "input_payload": json.dumps(retrieval_input),
                    "output_payload": json.dumps(retrieval.model_dump()),
                    "status": "completed",
                }
            )
        else:
            response_sections.append("No strong destination context was retrieved.")
            tool_logs.append(
                {
                    "tool_name": "destination_context_retriever",
                    "input_payload": json.dumps(retrieval_input),
                    "output_payload": "No relevant destination chunks were retrieved.",
                    "status": "completed",
                }
            )
        return {
            "response_sections": response_sections,
            "tool_logs": tool_logs,
        }
    except Exception as exc:
        logger.exception("RAG retrieval failed")
        tool_logs.append(
            {
                "tool_name": "destination_context_retriever",
                "input_payload": json.dumps(retrieval_input),
                "output_payload": f"RAG retrieval failed: {type(exc).__name__}.",
                "status": "failed",
            }
        )
        response_sections.append("Destination retrieval failed during this run.")
        return {
            "status": "partial" if status == "completed" else status,
            "response_sections": response_sections,
        "tool_logs": tool_logs,
    }


async def recommend_destinations_node(
    state: TripPlannerState, runtime: Runtime[TripPlannerRuntime]
) -> TripPlannerState:
    tool_logs = list(state["tool_logs"])
    response_sections = list(state["response_sections"])
    tool_registry = runtime.context.tool_registry
    tool_context = runtime.context.tool_context
    status = state["status"]
    travel_profile = state.get("travel_profile")

    if tool_registry is None or tool_context is None:
        tool_logs.append(
            {
                "tool_name": "destination_recommender",
                "input_payload": state["prompt"],
                "output_payload": "Destination recommendation failed because the tool runtime is unavailable.",
                "status": "failed",
            }
        )
        response_sections.append(
            "Destination recommendation could not run because the tool runtime is unavailable."
        )
        return {
            "status": "partial" if status == "completed" else status,
            "response_sections": response_sections,
            "tool_logs": tool_logs,
        }

    recommendation_input = DestinationRecommendationRequest(
        query_text=state["prompt"],
        budget_level=travel_profile.budget_level if travel_profile is not None else None,
        region=travel_profile.region if travel_profile is not None else None,
        limit=3,
    )

    try:
        recommendations = await tool_registry.get("destination_recommender").arun(
            recommendation_input,
            tool_context,
        )
    except Exception as exc:
        logger.exception("Destination recommendation failed")
        tool_logs.append(
            {
                "tool_name": "destination_recommender",
                "input_payload": json.dumps(recommendation_input.model_dump(mode="json")),
                "output_payload": (
                    f"Destination recommendation failed: {type(exc).__name__}."
                ),
                "status": "failed",
            }
        )
        response_sections.append("Destination recommendation failed during this run.")
        return {
            "status": "partial" if status == "completed" else status,
            "response_sections": response_sections,
            "tool_logs": tool_logs,
        }

    recommended_destinations = [
        recommendation.model_dump(mode="json")
        for recommendation in recommendations.results
    ]
    tool_logs.append(
        {
            "tool_name": "destination_recommender",
            "input_payload": json.dumps(recommendation_input.model_dump(mode="json")),
            "output_payload": json.dumps(recommendations.model_dump(mode="json")),
            "status": "completed",
        }
    )

    updates: dict[str, Any] = {
        "recommended_destinations": recommended_destinations,
        "tool_logs": tool_logs,
    }

    if recommended_destinations:
        top_destination = recommended_destinations[0]
        destination_summary = ", ".join(
            f"{item['destination']} ({item['score']})"
            for item in recommended_destinations
        )
        response_sections.append(f"Recommended destinations: {destination_summary}")
        selected_destination = (
            state.get("destination_name") or top_destination["destination"]
        )
        updates["destination_name"] = selected_destination
        if state.get("location_query") is None:
            generated_location_query = (
                f"{top_destination['destination']}, {top_destination['country']}"
            )
            updates["location_query"] = generated_location_query
            # The recommended destination is selected inside the graph, so any
            # previously inferred country code may no longer match it.
            updates["location_country_code"] = None
            response_sections.append(
                f"Generated weather lookup target from recommendation: {generated_location_query}"
            )
        response_sections.append(
            f"Primary recommendation selected for deeper analysis: {selected_destination}"
        )
    else:
        response_sections.append("No destination matches were found for this request.")

    updates["response_sections"] = response_sections
    return updates


async def live_conditions_node(
    state: TripPlannerState, runtime: Runtime[TripPlannerRuntime]
) -> TripPlannerState:
    tool_logs = list(state["tool_logs"])
    response_sections = list(state["response_sections"])
    tool_registry = runtime.context.tool_registry
    tool_context = runtime.context.tool_context
    status = state["status"]
    recommended_destinations = state.get("recommended_destinations") or []
    weather_location_query = state.get("location_query") or state.get("destination_name")

    if weather_location_query is None and recommended_destinations:
        top_destination = recommended_destinations[0]
        destination = top_destination.get("destination")
        country = top_destination.get("country")
        if destination and country:
            weather_location_query = f"{destination}, {country}"
        elif destination:
            weather_location_query = str(destination)

    if weather_location_query is None:
        response_sections.append(
            "Weather lookup target could not be determined from the prompt or recommendations."
        )
        tool_logs.append(
            {
                "tool_name": "live_conditions",
                "input_payload": "",
                "output_payload": "Live conditions skipped because no location query was provided.",
                "status": "skipped",
            }
        )
        return {
            "response_sections": response_sections,
            "tool_logs": tool_logs,
        }

    response_sections.append(
        f"Weather lookup target before live conditions: {weather_location_query}"
    )

    if tool_registry is None or tool_context is None:
        tool_logs.append(
            {
                "tool_name": "live_conditions",
                "input_payload": weather_location_query,
                "output_payload": (
                    "Live conditions could not run because shared services are unavailable."
                ),
                "status": "failed",
            }
        )
        response_sections.append(
            "Some live tool services were unavailable, so this run was only partially completed."
        )
        return {
            "status": "partial",
            "response_sections": response_sections,
            "tool_logs": tool_logs,
        }

    live_conditions_input = {
        "location_query": weather_location_query,
        "country_code": state.get("location_country_code"),
    }
    try:
        live_conditions = await tool_registry.get("live_conditions").arun(
            LiveConditionsRequest(**live_conditions_input),
            tool_context,
        )
        response_sections.append(
            "Current weather for "
            f"{live_conditions.location.name}: "
            f"{live_conditions.current.temperature_c:.1f}C, "
            f"{live_conditions.current.weather_summary}, "
            f"wind {live_conditions.current.wind_speed_kmh:.1f} km/h"
        )
        tool_logs.append(
            {
                "tool_name": "live_conditions",
                "input_payload": json.dumps(live_conditions_input),
                "output_payload": json.dumps(live_conditions.model_dump()),
                "status": "completed",
            }
        )
        return {
            "response_sections": response_sections,
            "tool_logs": tool_logs,
        }
    except Exception as exc:
        logger.exception("Live conditions lookup failed")
        tool_logs.append(
            {
                "tool_name": "live_conditions",
                "input_payload": json.dumps(live_conditions_input),
                "output_payload": (
                    f"Live conditions lookup failed: {type(exc).__name__}."
                ),
                "status": "failed",
            }
        )
        response_sections.append("Live conditions lookup failed during this run.")
        return {
            "status": "partial" if status == "completed" else status,
            "response_sections": response_sections,
            "tool_logs": tool_logs,
        }


async def synthesize_response_node(
    state: TripPlannerState, runtime: Runtime[TripPlannerRuntime]
) -> TripPlannerState:
    response_sections = list(state["response_sections"])
    destination_name = state.get("destination_name")
    recommended_destinations = state.get("recommended_destinations") or []
    tool_context = runtime.context.tool_context

    if destination_name is not None:
        response_sections.append(
            f"Use the retrieved context to evaluate {destination_name} for this trip."
        )
    if recommended_destinations:
        alternatives = [
            item["destination"] for item in recommended_destinations[1:3]
        ]
        if alternatives:
            response_sections.append(
                "Alternative options worth considering: " + ", ".join(alternatives)
            )

    if tool_context is None or tool_context.http_client is None:
        return {
            "response_sections": response_sections,
            "final_response": "\n".join(response_sections),
        }

    try:
        final_response = await synthesize_trip_response(
            tool_context.http_client,
            tool_context.settings,
            prompt=state["prompt"],
            destination_name=destination_name,
            response_sections=response_sections,
            tool_logs=state["tool_logs"],
        )
        return {
            "response_sections": response_sections,
            "final_response": final_response,
        }
    except LLMAuthenticationError:
        if tool_context.is_byok:
            raise
        response_sections.append(
            "LLM synthesis fallback used because the configured LLM API key was rejected."
        )
        return {
            "status": "partial" if state["status"] == "completed" else state["status"],
            "response_sections": response_sections,
            "final_response": "\n".join(response_sections),
        }
    except Exception as exc:
        response_sections.append(
            f"LLM synthesis fallback used because Claude generation failed: {type(exc).__name__}."
        )
        return {
            "status": "partial" if state["status"] == "completed" else state["status"],
            "response_sections": response_sections,
            "final_response": "\n".join(response_sections),
        }


@lru_cache(maxsize=1)
def build_trip_planner_graph():
    graph = StateGraph(TripPlannerState, context_schema=TripPlannerRuntime)
    graph.add_node("initialize", initialize_trip_state)
    graph.add_node("extract_request_fields", extract_request_fields_node)
    graph.add_node("recommend_destinations", recommend_destinations_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("live_conditions", live_conditions_node)
    graph.add_node("synthesize_response", synthesize_response_node)

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "extract_request_fields")
    graph.add_edge("extract_request_fields", "recommend_destinations")
    graph.add_edge("recommend_destinations", "retrieve_context")
    graph.add_edge("retrieve_context", "live_conditions")
    graph.add_edge("live_conditions", "synthesize_response")
    graph.add_edge("synthesize_response", END)

    return graph.compile()
