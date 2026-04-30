import json

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.tools.base import ToolContext
from app.agent.tools.registry import ToolRegistry
from app.db.models.agent_run import AgentRun
from app.db.models.user import User
from app.schemas.agent_runs import AgentRunCreate
from app.schemas.live_conditions import LiveConditionsRequest
from app.schemas.rag_retrieval import RagRetrievalRequest
from app.services.tool_logs import create_tool_log


async def create_agent_run(
    session: AsyncSession,
    current_user: User,
    payload: AgentRunCreate,
    *,
    tool_registry: ToolRegistry | None = None,
    tool_context: ToolContext | None = None,
) -> AgentRun:
    prompt = payload.prompt.strip()
    run_status = "completed"
    response_sections = [f"Prompt: {prompt}"]
    tool_log_payloads: list[dict[str, str]] = []

    predicted_style: str | None = None
    if payload.travel_profile is None:
        tool_log_payloads.append(
            {
                "tool_name": "travel_style_classifier",
                "input_payload": prompt,
                "output_payload": (
                    "Classifier skipped because no structured travel profile was provided."
                ),
                "status": "skipped",
            }
        )
    elif tool_registry is None or tool_context is None:
        run_status = "partial"
        tool_log_payloads.append(
            {
                "tool_name": "travel_style_classifier",
                "input_payload": json.dumps(payload.travel_profile.model_dump()),
                "output_payload": (
                    "Classifier could not run because the tool runtime is unavailable."
                ),
                "status": "failed",
            }
        )
        response_sections.append(
            "Travel style classification could not run because the tool runtime is unavailable."
        )
    else:
        try:
            prediction = await tool_registry.get("travel_style_classifier").arun(
                payload.travel_profile,
                tool_context,
            )
            predicted_style = prediction.predicted_style
            response_sections.append(f"Predicted travel style: {predicted_style}")
            tool_log_payloads.append(
                {
                    "tool_name": "travel_style_classifier",
                    "input_payload": json.dumps(payload.travel_profile.model_dump()),
                    "output_payload": json.dumps(prediction.model_dump()),
                    "status": "completed",
                }
            )
        except Exception as exc:
            run_status = "partial"
            tool_log_payloads.append(
                {
                    "tool_name": "travel_style_classifier",
                    "input_payload": json.dumps(payload.travel_profile.model_dump()),
                    "output_payload": (
                        f"Classifier failed: {type(exc).__name__}: {exc}"
                    ),
                    "status": "failed",
                }
            )
            response_sections.append("Travel style classification failed during this run.")

    if tool_registry is None or tool_context is None:
        run_status = "partial"
        tool_log_payloads.extend(
            [
                {
                    "tool_name": "destination_context_retriever",
                    "input_payload": prompt,
                    "output_payload": "RAG retrieval could not run because shared services are unavailable.",
                    "status": "failed",
                },
                {
                    "tool_name": "live_conditions",
                    "input_payload": payload.location_query or payload.destination_name or "",
                    "output_payload": "Live conditions could not run because shared services are unavailable.",
                    "status": "failed"
                    if (payload.location_query or payload.destination_name)
                    else "skipped",
                },
            ]
        )
        response_sections.append(
            "Some live tool services were unavailable, so this run was only partially completed."
        )
    else:
        try:
            retrieval = await tool_registry.get("destination_context_retriever").arun(
                RagRetrievalRequest(
                    query=prompt,
                    destination_name=payload.destination_name,
                    top_k=payload.retrieval_top_k,
                ),
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
                tool_log_payloads.append(
                    {
                        "tool_name": "destination_context_retriever",
                        "input_payload": json.dumps(
                            {
                                "query": prompt,
                                "destination_name": payload.destination_name,
                                "top_k": payload.retrieval_top_k,
                            }
                        ),
                        "output_payload": json.dumps(retrieval.model_dump()),
                        "status": "completed",
                    }
                )
            else:
                tool_log_payloads.append(
                    {
                        "tool_name": "destination_context_retriever",
                        "input_payload": json.dumps(
                            {
                                "query": prompt,
                                "destination_name": payload.destination_name,
                                "top_k": payload.retrieval_top_k,
                            }
                        ),
                        "output_payload": "No relevant destination chunks were retrieved.",
                        "status": "completed",
                    }
                )
                response_sections.append("No strong destination context was retrieved.")
        except Exception as exc:
            run_status = "partial"
            tool_log_payloads.append(
                {
                    "tool_name": "destination_context_retriever",
                    "input_payload": json.dumps(
                        {
                            "query": prompt,
                            "destination_name": payload.destination_name,
                            "top_k": payload.retrieval_top_k,
                        }
                    ),
                    "output_payload": f"RAG retrieval failed: {type(exc).__name__}: {exc}",
                    "status": "failed",
                }
            )
            response_sections.append("Destination retrieval failed during this run.")

        weather_location_query = payload.location_query or payload.destination_name
        if weather_location_query is None:
            tool_log_payloads.append(
                {
                    "tool_name": "live_conditions",
                    "input_payload": "",
                    "output_payload": "Live conditions skipped because no location query was provided.",
                    "status": "skipped",
                }
            )
        else:
            try:
                live_conditions = await tool_registry.get("live_conditions").arun(
                    LiveConditionsRequest(
                        location_query=weather_location_query,
                        country_code=payload.location_country_code,
                    ),
                    tool_context,
                )
                response_sections.append(
                    "Current weather for "
                    f"{live_conditions.location.name}: "
                    f"{live_conditions.current.temperature_c:.1f}C, "
                    f"{live_conditions.current.weather_summary}, "
                    f"wind {live_conditions.current.wind_speed_kmh:.1f} km/h"
                )
                tool_log_payloads.append(
                    {
                        "tool_name": "live_conditions",
                        "input_payload": json.dumps(
                            {
                                "location_query": weather_location_query,
                                "country_code": payload.location_country_code,
                            }
                        ),
                        "output_payload": json.dumps(live_conditions.model_dump()),
                        "status": "completed",
                    }
                )
            except Exception as exc:
                run_status = "partial"
                tool_log_payloads.append(
                    {
                        "tool_name": "live_conditions",
                        "input_payload": json.dumps(
                            {
                                "location_query": weather_location_query,
                                "country_code": payload.location_country_code,
                            }
                        ),
                        "output_payload": (
                            f"Live conditions lookup failed: {type(exc).__name__}: {exc}"
                        ),
                        "status": "failed",
                    }
                )
                response_sections.append("Live conditions lookup failed during this run.")

    if predicted_style is not None and payload.destination_name is not None:
        response_sections.append(
            f"Use the predicted style and retrieved context to evaluate {payload.destination_name} for this trip."
        )

    response = "\n".join(response_sections)

    agent_run = AgentRun(
        user_id=current_user.id,
        prompt=prompt,
        response=response,
        status=run_status,
    )
    session.add(agent_run)
    await session.commit()
    await session.refresh(agent_run)

    for tool_log_payload in tool_log_payloads:
        await create_tool_log(
            session,
            agent_run,
            tool_name=tool_log_payload["tool_name"],
            input_payload=tool_log_payload["input_payload"],
            output_payload=tool_log_payload["output_payload"],
            status=tool_log_payload["status"],
        )

    await session.refresh(agent_run, attribute_names=["tool_logs"])
    return agent_run
