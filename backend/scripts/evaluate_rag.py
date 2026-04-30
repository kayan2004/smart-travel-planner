import asyncio
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings
from app.db.session import create_db_engine, create_session_factory
from app.schemas.rag_retrieval import RagRetrievalRequest
from app.services.rag_retrieval import retrieve_destination_context

QUERY_FIXTURES_PATH = BACKEND_DIR / "data" / "rag_eval_queries.json"
ARTIFACT_DIR = BACKEND_DIR / "artifacts" / "rag"
JSON_REPORT_PATH = ARTIFACT_DIR / "rag_retrieval_eval.json"
CSV_REPORT_PATH = ARTIFACT_DIR / "rag_retrieval_eval.csv"


async def main() -> None:
    settings = get_settings()
    queries = _load_queries()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    engine = create_db_engine(settings)
    session_factory = create_session_factory(engine)
    http_client = httpx.AsyncClient(follow_redirects=True)

    try:
        async with session_factory() as session:
            detailed_results: list[dict[str, Any]] = []

            for index, item in enumerate(queries, start=1):
                query = item["query"]
                expected_destinations = item["expected_destinations"]
                print(f"[{index}/{len(queries)}] Evaluating: {query}")

                response = await retrieve_destination_context(
                    session,
                    http_client,
                    settings,
                    RagRetrievalRequest(query=query, top_k=5),
                )

                returned_destinations = [
                    result.destination_name for result in response.results
                ]
                matched_expected = any(
                    destination in expected_destinations
                    for destination in returned_destinations
                )
                top_result = response.results[0] if response.results else None

                detailed_results.append(
                    {
                        "query": query,
                        "expected_destinations": expected_destinations,
                        "matched_expected": matched_expected,
                        "top_result_destination": (
                            top_result.destination_name if top_result else None
                        ),
                        "top_result_similarity_score": (
                            top_result.similarity_score if top_result else None
                        ),
                        "results": [result.model_dump() for result in response.results],
                    }
                )

            _write_reports(detailed_results)
            print(f"Saved JSON report to {JSON_REPORT_PATH}")
            print(f"Saved CSV report to {CSV_REPORT_PATH}")
    finally:
        await http_client.aclose()
        await engine.dispose()


def _load_queries() -> list[dict[str, Any]]:
    return json.loads(QUERY_FIXTURES_PATH.read_text(encoding="utf-8"))


def _write_reports(detailed_results: list[dict[str, Any]]) -> None:
    JSON_REPORT_PATH.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "query_count": len(detailed_results),
                "results": detailed_results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with CSV_REPORT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "query",
                "expected_destinations",
                "matched_expected",
                "top_result_destination",
                "top_result_similarity_score",
            ],
        )
        writer.writeheader()
        for item in detailed_results:
            writer.writerow(
                {
                    "query": item["query"],
                    "expected_destinations": ", ".join(item["expected_destinations"]),
                    "matched_expected": item["matched_expected"],
                    "top_result_destination": item["top_result_destination"],
                    "top_result_similarity_score": item["top_result_similarity_score"],
                }
            )


if __name__ == "__main__":
    asyncio.run(main())
