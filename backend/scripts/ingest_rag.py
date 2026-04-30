import asyncio
import sys
from pathlib import Path

import httpx

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings
from app.db.session import create_db_engine, create_session_factory
from app.services.rag_ingestion import ingest_destination_documents


async def main() -> None:
    settings = get_settings()
    engine = create_db_engine(settings)
    session_factory = create_session_factory(engine)
    http_client = httpx.AsyncClient(follow_redirects=True)

    try:
        async with session_factory() as session:
            result = await ingest_destination_documents(session, http_client, settings)
            print(result)
    finally:
        await http_client.aclose()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
